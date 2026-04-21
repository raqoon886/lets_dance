import os
import sys
import numpy as np
import onnxruntime as ort

class DanceONNXScorer:
    """
    Handles ONNX model loading, preprocessing, inference, and score computation
    for the dance embedding ONNX pipeline.
    """
    def __init__(self, model_path: str, window_size: int = 64, out_dim: int = 64):
        """
        Args:
            model_path: Path to the ONNX model.
            window_size: Expected window size frames.
            out_dim: Expected output dimension. Will slice if model outputs more.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Check input info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Check output info
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        
        self.window_size = window_size
        self.out_dim = out_dim
        
        print(f"[ONNXScorer] Loaded model {os.path.basename(model_path)}")
        print(f"[ONNXScorer] Input shape: {self.input_shape}, Raw Output shape: {self.output_shape}")
        print(f"[ONNXScorer] Final Output Dimension config: {self.out_dim}")
        
    def get_embedding(self, sequence: np.ndarray) -> np.ndarray:
        """
        Runs inference on a prepared sequence tensor.
        
        Args:
            sequence: Numpy array of shape [1, 3, window_size, 17]
        Returns: 
            Embedding array (e.g., shape [1, out_dim])
        """
        if len(sequence.shape) != 4:
            raise ValueError(f"Expected sequence shape [1, 3, {self.window_size}, 17], got {sequence.shape}")
            
        if sequence.shape[2] != self.window_size:
            print(f"Warning: Sequence window ({sequence.shape[2]}) does not match model window_size ({self.window_size})")

        # Run ONNX inference
        ort_inputs = {self.input_name: sequence.astype(np.float32)}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        
        embedding = ort_outs[0]
        
        # Enforce exactly out_dim (64)
        if embedding.shape[-1] > self.out_dim:
            embedding = embedding[..., :self.out_dim]
            
        # Re-normalize after slicing to keep it unit length for cosine similarity
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / np.maximum(norm, 1e-8)
        
        return embedding

    def compute_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine similarity between two embeddings and returns a 0-100 score.
        """
        emb1_flat = emb1.flatten()
        emb2_flat = emb2.flatten()
        
        # Cosine similarity
        num = np.dot(emb1_flat, emb2_flat)
        den = np.linalg.norm(emb1_flat) * np.linalg.norm(emb2_flat)
        
        similarity = num / max(den, 1e-8)
        
        # Convert -1.0~1.0 to 0~100 scale.
        # Alternatively, max(0.0, float(similarity)) * 100.0 can be used.
        score = max(0.0, float(similarity)) * 100.0
        return score
