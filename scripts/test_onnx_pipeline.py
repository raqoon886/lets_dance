import os
import sys
import numpy as np

# Add project root to sys path to allow importing src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference.onnx_scorer import DanceONNXScorer

def main():
    # Model path relative to the script execution dir (expecting project root)
    model_path = os.path.join(project_root, "data", "models", "scratch", "dance_embedding.onnx")
    
    print(f"Initializing Scorer with model: {model_path}")
    
    # Initialize scorer
    try:
        scorer = DanceONNXScorer(model_path=model_path, window_size=64)
    except Exception as e:
        print(f"Failed to load scorer: {e}")
        return

    print(f"\n--- Testing Inference Pipeline ---")
    
    # Create dummy input: Batch=1, Channels=3, Window=64, Keypoints=17
    # 3 could be (x, y, confidence)
    dummy_input_1 = np.random.randn(1, 3, 64, 17).astype(np.float32)
    # Slightly altered input for testing
    dummy_input_2 = dummy_input_1 + (np.random.randn(1, 3, 64, 17).astype(np.float32) * 0.1)

    print(f"Feeding input of shape: {dummy_input_1.shape}")

    # Get embeddings
    emb1 = scorer.get_embedding(dummy_input_1)
    emb2 = scorer.get_embedding(dummy_input_2)

    print(f"Embed 1 shape: {emb1.shape}")
    print(f"Embed 2 shape: {emb2.shape}")
    
    # Explicitly check if output dimension is 64
    output_dim = emb1.shape[-1]
    if output_dim == 64:
        print("✅ Output dimension is exactly 64!")
    else:
        print(f"❌ Output dimension is {output_dim}, expected 64.")

    # Compute score
    score = scorer.compute_score(emb1, emb2)
    print(f"\n🎯 Output Score (Cosine Similarity scaled to 0-100): {score:.2f}")

if __name__ == "__main__":
    main()
