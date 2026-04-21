"""Scratch TFLite model based pose similarity scoring."""

from collections import deque
import json
import os

import numpy as np

from pose.landmark_utils import DANCE_JOINTS
from scoring.scratch_features import build_pose_window, normalize_pose_landmarks


def resolve_scratch_model_path(model_name=None, model_dir="data/models/scratch",
                               model_path=None, registry_name="model_registry.json"):
    """Resolve a scratch model name to a .tflite path using the registry."""
    if model_path:
        path = model_path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    model_name = model_name or "gcn_wide"
    model_dir = os.path.abspath(model_dir)
    registry_path = os.path.join(model_dir, registry_name)
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        for item in registry.get("variants", []):
            if item.get("name") == model_name:
                path = item.get("paths", {}).get("tflite")
                if not path:
                    break
                return path if os.path.isabs(path) else os.path.abspath(path)

    fallback = os.path.join(model_dir, f"{model_name}.tflite")
    return fallback


class ScratchPoseSimilarity:
    """Compare pose windows with a scratch-trained TFLite encoder model."""

    LAYOUTS = ("BTJC", "BJTC", "BTC")

    def __init__(self, model_path: str, sequence_length=30, feature_dims=2,
                 input_layout="BTJC", target_joints=None, top_k=3,
                 candidate_stride=3, interpreter=None, similarity_threshold=0.70):
        if input_layout not in self.LAYOUTS:
            raise ValueError(f"input_layout must be one of {self.LAYOUTS}: {input_layout}")

        self.model_path = model_path
        self.sequence_length = int(sequence_length)
        self.feature_dims = int(feature_dims)
        self.input_layout = input_layout
        self.target_joints = list(target_joints or DANCE_JOINTS)
        self.top_k = max(1, int(top_k))
        self.candidate_stride = max(1, int(candidate_stride))
        self.similarity_threshold = float(similarity_threshold)

        # Ring buffer — deque + np.stack 대비 메모리 복사 최소화
        self._ring_buffer = None   # lazy init (shape unknown until first frame)
        self._ring_idx = 0
        self._ring_count = 0
        self._ref_embedding_cache = {}
        self._interpreter = interpreter
        self._input_details = None
        self._output_details = None
        self._precomputed_ref_embeddings = None
        self._precomputed_ref_index_to_pos = {}
        self._precomputed_ref_meta = None
        self._precomputed_ref_path = None
        self._last_debug_info = None

        if interpreter is None:
            self._load_interpreter()
        else:
            self._refresh_io_details()

    def reset(self):
        """Clear rolling user window and cached reference embeddings."""
        self._ring_buffer = None
        self._ring_idx = 0
        self._ring_count = 0
        self._ref_embedding_cache.clear()
        self.clear_reference_embedding_cache()
        self._last_debug_info = None

    def clear_reference_embedding_cache(self):
        """Clear the disk-backed reference embedding cache for the current song."""
        self._precomputed_ref_embeddings = None
        self._precomputed_ref_index_to_pos = {}
        self._precomputed_ref_meta = None
        self._precomputed_ref_path = None

    def compute(self, user_landmarks: np.ndarray, reference_sequence: np.ndarray,
                reference_index: int, tolerance_frames: int = 0,
                timing_penalty: float = 0.0):
        """Return a finite similarity in [0, 1], or None until a full window exists.

        This method combines buffer_frame() + compute_from_buffer() for
        backward compatibility.  New callers should prefer the split API.
        """
        if user_landmarks is None or reference_sequence is None:
            return None

        self.buffer_frame(user_landmarks)
        return self.compute_from_buffer(reference_sequence, reference_index,
                                        tolerance_frames=tolerance_frames,
                                        timing_penalty=timing_penalty)

    # ── split API: buffer accumulation vs. inference ──────────────

    def buffer_frame(self, user_landmarks: np.ndarray):
        """Normalize and append one frame to the rolling user window.

        Call this every frame to keep the sliding window up-to-date,
        even when inference is not needed.
        """
        if user_landmarks is None:
            return
        normalized = normalize_pose_landmarks(
            user_landmarks,
            target_joints=self.target_joints,
            feature_dims=self.feature_dims,
        )
        if self._ring_buffer is None:
            J, C = normalized.shape
            self._ring_buffer = np.zeros(
                (self.sequence_length, J, C), dtype=np.float32)
        self._ring_buffer[self._ring_idx] = normalized
        self._ring_idx = (self._ring_idx + 1) % self.sequence_length
        self._ring_count = min(self._ring_count + 1, self.sequence_length)

    def compute_from_buffer(self, reference_sequence: np.ndarray,
                            reference_index: int, tolerance_frames: int = 0,
                            timing_penalty: float = 0.0):
        """Run model inference on the current buffer and return similarity.

        Returns None if the buffer is not yet full (< sequence_length frames).
        """
        if reference_sequence is None:
            return None
        if self._ring_count < self.sequence_length:
            return None

        # Ring buffer → chronological order (single copy via np.roll)
        user_window = np.roll(self._ring_buffer, -self._ring_idx, axis=0)
        candidate_indices = self._candidate_reference_indices(
            len(reference_sequence), reference_index, tolerance_frames)
        if not candidate_indices:
            return None

        user_embedding = self._normalize_vector(self._infer_single(user_window))
        ref_embeddings = self._reference_embeddings(reference_sequence, candidate_indices)
        if ref_embeddings is not None:
            sims = np.dot(ref_embeddings, user_embedding)
        else:
            sims = [
                self._cosine_similarity(user_embedding, self._reference_embedding(
                    reference_sequence, end_idx))
                for end_idx in candidate_indices
            ]

        scored_candidates = self._apply_timing_penalty(
            sims,
            candidate_indices,
            reference_index,
            tolerance_frames,
            timing_penalty,
            self.similarity_threshold,
        )
        if not scored_candidates:
            self._last_debug_info = {
                "current_reference_index": int(reference_index),
                "candidate_indices": [int(idx) for idx in candidate_indices],
                "best_candidate_index": None,
                "best_candidate_similarity": 0.0,
                "returned_similarity": 0.0,
            }
            return 0.0
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        top_candidates = scored_candidates[:min(self.top_k, len(scored_candidates))]
        returned_similarity = float(np.mean([score for _, score in top_candidates]))
        best_candidate_index, best_candidate_similarity = top_candidates[0]
        self._last_debug_info = {
            "current_reference_index": int(reference_index),
            "candidate_indices": [int(idx) for idx in candidate_indices],
            "best_candidate_index": int(best_candidate_index),
            "best_candidate_similarity": float(best_candidate_similarity),
            "returned_similarity": returned_similarity,
            "top_candidates": [
                {"reference_index": int(idx), "similarity": float(score)}
                for idx, score in top_candidates
            ],
        }
        return returned_similarity

    @staticmethod
    def _apply_timing_penalty(sims, candidate_indices, reference_index,
                              tolerance_frames, timing_penalty, similarity_threshold=0.0):
        """Scale similarities and penalize matches far from the current beat."""
        timing_penalty = max(0.0, float(timing_penalty or 0.0))
        tolerance_frames = max(0, int(tolerance_frames or 0))
        scored = []
        for sim, end_idx in zip(np.asarray(sims).reshape(-1), candidate_indices):
            if not np.isfinite(sim):
                continue
                
            adjusted = float(sim)
            
            # 1. Similarity Rescaling: expand discriminative range using threshold
            if similarity_threshold > 0.0:
                if adjusted <= similarity_threshold:
                    adjusted = 0.0
                elif similarity_threshold < 1.0:
                    adjusted = (adjusted - similarity_threshold) / (1.0 - similarity_threshold)

            # 2. Timing Penalty: non-linear (quadratic) decay within tolerance
            if timing_penalty > 0.0 and tolerance_frames > 0:
                offset = abs(int(reference_index) - int(end_idx))
                penalty_ratio = min(offset / tolerance_frames, 1.0)
                adjusted -= timing_penalty * (penalty_ratio ** 2)
                
            scored.append((int(end_idx), float(np.clip(adjusted, 0.0, 1.0))))
        return scored

    def _load_interpreter(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Scratch TFLite model not found: {self.model_path}")
        Interpreter, backend = self._resolve_interpreter()

        try:
            self._interpreter = Interpreter(model_path=self.model_path)
        except ValueError as exc:
            if backend == "tflite-runtime" and "Didn't find op" in str(exc):
                raise ValueError(
                    "The installed tflite-runtime package is too old for this "
                    "TFLite model. Install ai-edge-litert, or re-export the model "
                    "with the same TensorFlow/TFLite version used by the runtime."
                ) from exc
            raise

        self._resize_dynamic_input()
        self._interpreter.allocate_tensors()
        self._refresh_io_details()

    @staticmethod
    def _resolve_interpreter():
        try:
            from ai_edge_litert.interpreter import Interpreter
            return Interpreter, "ai-edge-litert"
        except ImportError:
            pass

        try:
            import tensorflow as tf
            return tf.lite.Interpreter, "tensorflow"
        except ImportError:
            pass

        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter, "tflite-runtime"
        except ImportError as exc:
            raise ImportError(
                "Scratch scoring requires ai-edge-litert, tensorflow, or "
                "tflite-runtime. Install ai-edge-litert for the lightweight "
                "runtime, or tensorflow for training/dev."
            ) from exc

    def _resize_dynamic_input(self):
        details = self._interpreter.get_input_details()
        if not details:
            return
        shape_sig = details[0].get("shape_signature", details[0].get("shape"))
        if shape_sig is not None and np.any(np.array(shape_sig) < 0):
            self._interpreter.resize_tensor_input(
                details[0]["index"], self._expected_input_shape(), strict=False)

    def _refresh_io_details(self):
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def _candidate_reference_indices(self, total_frames, reference_index, tolerance_frames):
        end = min(int(reference_index), int(total_frames) - 1)
        if end < self.sequence_length - 1:
            return []
        start = max(self.sequence_length - 1, end - int(tolerance_frames))
        
        # 캐시 히트율(Cache Hit Rate) 극대화를 위해 start를 candidate_stride의 배수로 정렬
        remainder = start % self.candidate_stride
        start_aligned = start if remainder == 0 else start + (self.candidate_stride - remainder)
        
        indices = list(range(start_aligned, end + 1, self.candidate_stride))
        if not indices or indices[-1] != end:
            indices.append(end)
        return indices

    def warmup_reference_embeddings(self, reference_sequence):
        """Pre-compute reference embeddings (background thread friendly)."""
        if reference_sequence is None:
            return
        if self._precomputed_ref_embeddings is not None:
            return
            
        total_frames = len(reference_sequence)
        start = self.sequence_length - 1
        remainder = start % self.candidate_stride
        start_aligned = start if remainder == 0 else start + (self.candidate_stride - remainder)
        
        import threading
        
        def _task():
            try:
                # 독립적인 인터프리터를 가져 스레드 충돌(Crash)을 방지
                temp_encoder = ScratchPoseSimilarity(
                    model_path=self.model_path,
                    sequence_length=self.sequence_length,
                    feature_dims=self.feature_dims,
                    input_layout=self.input_layout,
                    target_joints=self.target_joints,
                    top_k=self.top_k,
                    candidate_stride=self.candidate_stride,
                    similarity_threshold=self.similarity_threshold,
                )
                # 원본 캐시 참조(공유). Python dict 삽입은 Thread-safe
                temp_encoder._ref_embedding_cache = self._ref_embedding_cache
                
                for idx in range(start_aligned, total_frames, self.candidate_stride):
                    # 만약 게임 메인 루프에서 먼저 계산했다면 건너뜀
                    if idx not in temp_encoder._ref_embedding_cache:
                        temp_encoder._reference_embedding(reference_sequence, idx)
                print("[INFO] TFLite background warmup complete.")
            except Exception as e:
                print(f"[WARN] TFLite warmup failed: {e}")
                
        threading.Thread(target=_task, daemon=True, name="tflite-warmup").start()

    def load_reference_embedding_cache(self, cache_dir, dance_name, model_kind,
                                       model_name, reference_path=None):
        """Load a precomputed per-song reference embedding .npz cache.

        Returns True when a compatible cache was loaded. If no cache is found or
        metadata does not match the current encoder settings, returns False and
        leaves the comparator in fallback mode.
        """
        self.clear_reference_embedding_cache()
        cache_dir = os.path.abspath(cache_dir)
        if not os.path.isdir(cache_dir):
            return False

        prefix = (
            f"{dance_name}__{model_kind}__{model_name}"
            f"__seq{self.sequence_length}_fd{self.feature_dims}_{self.input_layout}_stride"
        )
        candidates = []
        for filename in os.listdir(cache_dir):
            if not filename.startswith(prefix) or not filename.endswith(".npz"):
                continue
            path = os.path.join(cache_dir, filename)
            try:
                with np.load(path, allow_pickle=True) as data:
                    meta = json.loads(str(data["metadata"]))
                stride = int(meta.get("cache_stride", 10**9))
                candidates.append((stride, path, meta))
            except Exception as exc:
                print(f"[WARN] Invalid reference embedding cache ignored: {path} ({exc})")

        if not candidates:
            return False

        candidates.sort(key=lambda item: item[0])
        for _, path, meta in candidates:
            if not self._is_compatible_reference_cache(
                    meta, dance_name, model_kind, model_name, reference_path):
                continue
            try:
                with np.load(path, allow_pickle=True) as data:
                    end_indices = data["end_indices"].astype(np.int32)
                    embeddings = data["embeddings"].astype(np.float32)
                    if "metadata" in data:
                        meta = json.loads(str(data["metadata"]))
                if embeddings.ndim != 2 or len(end_indices) != len(embeddings):
                    raise ValueError(
                        f"Invalid shapes: end_indices={end_indices.shape}, "
                        f"embeddings={embeddings.shape}"
                    )
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
                self._precomputed_ref_embeddings = embeddings.astype(np.float32)
                self._precomputed_ref_index_to_pos = {
                    int(end_idx): int(pos) for pos, end_idx in enumerate(end_indices)
                }
                self._precomputed_ref_meta = meta
                self._precomputed_ref_path = path
                self._ref_embedding_cache.clear()
                print(
                    "[INFO] Reference embedding cache loaded: "
                    f"{os.path.basename(path)} "
                    f"({embeddings.shape[0]} windows, dim={embeddings.shape[1]})"
                )
                return True
            except Exception as exc:
                print(f"[WARN] Failed to load reference embedding cache: {path} ({exc})")
        return False

    def _is_compatible_reference_cache(self, meta, dance_name, model_kind,
                                       model_name, reference_path=None):
        if meta.get("dance_name") != dance_name:
            return False
        if meta.get("model_kind") != model_kind:
            return False
        if meta.get("model_name") != model_name:
            return False
        if int(meta.get("sequence_length", -1)) != self.sequence_length:
            return False
        if int(meta.get("feature_dims", -1)) != self.feature_dims:
            return False
        if meta.get("input_layout") != self.input_layout:
            return False
        cached_joints = [int(j) for j in meta.get("target_joints", [])]
        if cached_joints and cached_joints != [int(j) for j in self.target_joints]:
            return False
        if reference_path:
            cached_path = meta.get("reference_path")
            if cached_path and os.path.basename(cached_path) != os.path.basename(reference_path):
                return False
        return True

    def _reference_window(self, reference_sequence, end_idx):
        return build_pose_window(
            reference_sequence,
            end_idx,
            self.sequence_length,
            target_joints=self.target_joints,
            feature_dims=self.feature_dims,
        )

    def _reference_embedding(self, reference_sequence, end_idx):
        cached = self._lookup_precomputed_reference_embedding(end_idx)
        if cached is not None:
            return cached
        cached = self._ref_embedding_cache.get(end_idx)
        if cached is not None:
            return cached
        embedding = self._normalize_vector(
            self._infer_single(self._reference_window(reference_sequence, end_idx)))
        self._ref_embedding_cache[end_idx] = embedding
        return embedding

    def _reference_embeddings(self, reference_sequence, end_indices):
        if self._precomputed_ref_embeddings is None:
            return None
        positions = []
        for end_idx in end_indices:
            pos = self._precomputed_ref_index_to_pos.get(int(end_idx))
            if pos is None:
                return None
            positions.append(pos)
        return self._precomputed_ref_embeddings[np.asarray(positions, dtype=np.int32)]

    def _lookup_precomputed_reference_embedding(self, end_idx):
        if self._precomputed_ref_embeddings is None:
            return None
        pos = self._precomputed_ref_index_to_pos.get(int(end_idx))
        if pos is None:
            return None
        return self._precomputed_ref_embeddings[pos]

    def _infer_single(self, sequence):
        detail = self._input_details[0]
        tensor = self._format_input(sequence)
        tensor = self._fit_to_shape(tensor, detail)
        tensor = self._quantize_if_needed(tensor, detail)
        self._interpreter.set_tensor(detail["index"], tensor)
        self._interpreter.invoke()
        return self._get_output_vector()

    def _expected_input_shape(self):
        joints = len(self.target_joints)
        if self.input_layout == "BTJC":
            return np.array([1, self.sequence_length, joints, self.feature_dims], dtype=np.int32)
        if self.input_layout == "BJTC":
            return np.array([1, joints, self.sequence_length, self.feature_dims], dtype=np.int32)
        return np.array([1, self.sequence_length, joints * self.feature_dims], dtype=np.int32)

    def _format_input(self, sequence):
        seq = np.asarray(sequence, dtype=np.float32)
        if self.input_layout == "BTJC":
            tensor = seq[None, ...]
        elif self.input_layout == "BJTC":
            tensor = np.transpose(seq, (1, 0, 2))[None, ...]
        else:
            tensor = seq.reshape(self.sequence_length, -1)[None, ...]
        return tensor.astype(np.float32)

    @staticmethod
    def _fit_to_shape(tensor, detail):
        expected = np.array(detail.get("shape", []), dtype=np.int32)
        if expected.size and np.all(expected > 0) and tuple(expected) != tensor.shape:
            raise ValueError(
                f"Scratch model input shape mismatch: expected {tuple(expected)}, "
                f"got {tensor.shape}"
            )
        return tensor

    @staticmethod
    def _quantize_if_needed(tensor, detail):
        dtype = detail.get("dtype", np.float32)
        if np.issubdtype(dtype, np.floating):
            return tensor.astype(dtype)
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if not scale:
            return tensor.astype(dtype)
        quantized = np.round(tensor / scale + zero_point)
        info = np.iinfo(dtype)
        return np.clip(quantized, info.min, info.max).astype(dtype)

    def _get_output_vector(self):
        detail = self._output_details[0]
        out = self._interpreter.get_tensor(detail["index"])
        dtype = detail.get("dtype", out.dtype)
        if not np.issubdtype(dtype, np.floating):
            scale, zero_point = detail.get("quantization", (0.0, 0))
            if scale:
                out = (out.astype(np.float32) - zero_point) * scale
        return np.nan_to_num(np.asarray(out, dtype=np.float32).reshape(-1))

    @staticmethod
    def _cosine_similarity(a, b):
        a = ScratchPoseSimilarity._normalize_vector(a)
        b = ScratchPoseSimilarity._normalize_vector(b)
        if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
            return 0.0
        return float(np.dot(a, b))

    @staticmethod
    def _normalize_vector(vector):
        vector = np.nan_to_num(np.asarray(vector, dtype=np.float32).reshape(-1))
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return vector
        return vector / norm
