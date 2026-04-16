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
                 candidate_stride=3, interpreter=None):
        if input_layout not in self.LAYOUTS:
            raise ValueError(f"input_layout must be one of {self.LAYOUTS}: {input_layout}")

        self.model_path = model_path
        self.sequence_length = int(sequence_length)
        self.feature_dims = int(feature_dims)
        self.input_layout = input_layout
        self.target_joints = list(target_joints or DANCE_JOINTS)
        self.top_k = max(1, int(top_k))
        self.candidate_stride = max(1, int(candidate_stride))

        self._user_buffer = deque(maxlen=self.sequence_length)
        self._ref_embedding_cache = {}
        self._interpreter = interpreter
        self._input_details = None
        self._output_details = None

        if interpreter is None:
            self._load_interpreter()
        else:
            self._refresh_io_details()

    def reset(self):
        """Clear rolling user window and cached reference embeddings."""
        self._user_buffer.clear()
        self._ref_embedding_cache.clear()

    def compute(self, user_landmarks: np.ndarray, reference_sequence: np.ndarray,
                reference_index: int, tolerance_frames: int = 0):
        """Return a finite similarity in [0, 1], or None until a full window exists."""
        if user_landmarks is None or reference_sequence is None:
            return None

        self._user_buffer.append(normalize_pose_landmarks(
            user_landmarks,
            target_joints=self.target_joints,
            feature_dims=self.feature_dims,
        ))
        if len(self._user_buffer) < self.sequence_length:
            return None

        user_window = np.stack(list(self._user_buffer), axis=0).astype(np.float32)
        candidate_indices = self._candidate_reference_indices(
            len(reference_sequence), reference_index, tolerance_frames)
        if not candidate_indices:
            return None

        user_embedding = self._infer_single(user_window)
        sims = [
            self._cosine_similarity(user_embedding, self._reference_embedding(
                reference_sequence, end_idx))
            for end_idx in candidate_indices
        ]
        sims = [float(np.clip(s, 0.0, 1.0)) for s in sims if np.isfinite(s)]
        if not sims:
            return 0.0
        sims.sort(reverse=True)
        return float(np.mean(sims[:min(self.top_k, len(sims))]))

    def _load_interpreter(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Scratch TFLite model not found: {self.model_path}")
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError as exc:
                raise ImportError(
                    "Scratch scoring requires tflite-runtime or tensorflow. "
                    "Install tflite-runtime for runtime, or tensorflow for training/dev."
                ) from exc

        self._interpreter = Interpreter(model_path=self.model_path)
        self._resize_dynamic_input()
        self._interpreter.allocate_tensors()
        self._refresh_io_details()

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

    def _reference_window(self, reference_sequence, end_idx):
        return build_pose_window(
            reference_sequence,
            end_idx,
            self.sequence_length,
            target_joints=self.target_joints,
            feature_dims=self.feature_dims,
        )

    def _reference_embedding(self, reference_sequence, end_idx):
        cached = self._ref_embedding_cache.get(end_idx)
        if cached is not None:
            return cached
        embedding = self._infer_single(self._reference_window(reference_sequence, end_idx))
        self._ref_embedding_cache[end_idx] = embedding
        return embedding

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
        a = np.nan_to_num(np.asarray(a, dtype=np.float32))
        b = np.nan_to_num(np.asarray(b, dtype=np.float32))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
