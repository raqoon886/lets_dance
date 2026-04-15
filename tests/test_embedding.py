"""Tests for embedding extraction pipeline."""

import numpy as np
import pytest


class TestSkeletonGraphBuilder:
    """Test suite for SkeletonGraphBuilder."""

    def test_edge_index_shape(self):
        """Test edge index is bidirectional."""
        from embedding.graph_builder import SkeletonGraphBuilder
        builder = SkeletonGraphBuilder()
        edge_index = builder._edge_index
        assert edge_index.shape[0] == 2
        # Bidirectional: 2x the number of edges
        assert edge_index.shape[1] == len(builder.ADJACENCY) * 2

    def test_build_frame_graph(self):
        """Test single frame graph construction."""
        from embedding.graph_builder import SkeletonGraphBuilder
        builder = SkeletonGraphBuilder()
        landmarks = np.random.randn(12, 3).astype(np.float32)
        graph = builder.build_frame_graph(landmarks)
        assert "node_features" in graph
        assert "edge_index" in graph
        assert graph["node_features"].shape == (12, 3)

    def test_build_sequence_graph(self):
        """Test sequence graph with temporal edges."""
        from embedding.graph_builder import SkeletonGraphBuilder
        builder = SkeletonGraphBuilder()
        sequence = [np.random.randn(12, 3) for _ in range(10)]
        graph = builder.build_sequence_graph(sequence)
        assert graph["sequence_length"] == 10
        assert graph["node_features"].shape[0] == 10 * 12


class TestDanceEmbeddingModel:
    """Test suite for DanceEmbeddingModel."""

    def test_forward_output_shape(self):
        """Test model output has correct embedding dimension."""
        from embedding.model import DanceEmbeddingModel
        model = DanceEmbeddingModel(embedding_dim=128)
        graph = {"node_features": np.zeros((12, 3)), "edge_index": np.zeros((2, 0))}
        embedding = model.forward(graph)
        assert embedding.shape == (128,)
