"""Tests for pose detection and landmark processing."""

import numpy as np
import pytest


class TestPoseDetector:
    """Test suite for PoseDetector."""

    def test_initialize(self):
        """Test detector initialization."""
        from pose.detector import PoseDetector
        detector = PoseDetector()
        detector.initialize()
        # TODO: Assert model is loaded

    def test_detect_returns_correct_shape(self):
        """Test that detect returns landmarks with shape (33, 4)."""
        from pose.detector import PoseDetector
        detector = PoseDetector()
        detector.initialize()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result["landmarks"].shape == (33, 4)
        assert isinstance(result["detected"], bool)

    def test_release(self):
        """Test resource cleanup."""
        from pose.detector import PoseDetector
        detector = PoseDetector()
        detector.initialize()
        detector.release()


class TestLandmarkProcessor:
    """Test suite for LandmarkProcessor."""

    def test_normalize_output_shape(self):
        """Test normalized output has correct shape."""
        from pose.landmark_utils import LandmarkProcessor
        processor = LandmarkProcessor()
        landmarks = np.random.randn(33, 4).astype(np.float32)
        normalized = processor.normalize(landmarks)
        assert normalized.shape == (len(processor.target_joints), 3)

    def test_compute_joint_angles(self):
        """Test joint angle computation."""
        from pose.landmark_utils import LandmarkProcessor
        processor = LandmarkProcessor()
        landmarks = np.random.randn(len(processor.target_joints), 3)
        angles = processor.compute_joint_angles(landmarks)
        assert len(angles) == len(processor.target_joints)

    def test_smooth_empty_buffer(self):
        """Test smoothing with empty buffer returns zeros."""
        from pose.landmark_utils import LandmarkProcessor
        processor = LandmarkProcessor()
        result = processor.smooth([])
        assert result.shape == (len(processor.target_joints), 3)
