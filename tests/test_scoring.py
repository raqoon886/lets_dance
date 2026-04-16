"""Tests for scoring and feedback systems."""

import numpy as np
import pytest


class TestSimilarityCalculator:
    """Test suite for SimilarityCalculator."""

    def test_cosine_metric(self):
        """Test cosine similarity returns value in [0, 1]."""
        from scoring.similarity import SimilarityCalculator
        calc = SimilarityCalculator(metric="cosine")
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        score = calc.compute(a, b)
        assert isinstance(score, float)

    def test_compute_sequence(self):
        """Test sequence similarity computation."""
        from scoring.similarity import SimilarityCalculator
        calc = SimilarityCalculator()
        user = [np.random.randn(128) for _ in range(10)]
        ref = [np.random.randn(128) for _ in range(10)]
        scores = calc.compute_sequence(user, ref)
        assert len(scores) == 10


class TestDanceScorer:
    """Test suite for DanceScorer."""

    def test_evaluate_returns_correct_keys(self):
        """Test evaluation result has all required keys."""
        from scoring.scorer import DanceScorer
        scorer = DanceScorer()
        result = scorer.evaluate(0.85)
        assert "grade" in result
        assert "points" in result
        assert "combo" in result
        assert "total_score" in result

    def test_combo_tracking(self):
        """Test combo increments on non-miss and resets on miss."""
        from scoring.scorer import DanceScorer
        scorer = DanceScorer()
        scorer.evaluate(0.95)  # Perfect
        scorer.evaluate(0.80)  # Great
        assert scorer.combo == 2
        scorer.evaluate(0.10)  # Miss
        assert scorer.combo == 0

    def test_reset(self):
        """Test score reset."""
        from scoring.scorer import DanceScorer
        scorer = DanceScorer()
        scorer.evaluate(0.9)
        scorer.reset()
        assert scorer.total_score == 0
        assert scorer.combo == 0

    def test_final_result(self):
        """Test final result compilation."""
        from scoring.scorer import DanceScorer
        scorer = DanceScorer()
        for sim in [0.95, 0.80, 0.65, 0.30, 0.90]:
            scorer.evaluate(sim)
        result = scorer.get_final_result()
        assert "total_score" in result
        assert "max_combo" in result
        assert result["total_moves"] == 5


class TestFeedbackGenerator:
    """Test suite for FeedbackGenerator."""

    def test_generate_feedback(self):
        """Test feedback generation from evaluation."""
        from scoring.feedback import FeedbackGenerator
        gen = FeedbackGenerator()
        evaluation = {"grade": "Perfect", "combo": 5, "points": 150}
        feedback = gen.generate(evaluation)
        assert feedback["text"] == "PERFECT!"
        assert feedback["combo"] == 5

    def test_feedback_timer(self):
        """Test feedback display timer."""
        from scoring.feedback import FeedbackGenerator
        gen = FeedbackGenerator()
        gen.generate({"grade": "Great", "combo": 1, "points": 80})
        assert gen.is_active
        for _ in range(100):
            gen.update()
        assert not gen.is_active


class TestScratchPoseSimilarity:
    """Test scratch TFLite scoring wrapper without a real TFLite file."""

    class FakeInterpreter:
        def __init__(self):
            self._input = None

        def get_input_details(self):
            return [{
                "index": 0,
                "shape": np.array([1, 3, 12, 2], dtype=np.int32),
                "dtype": np.float32,
            }]

        def get_output_details(self):
            return [{
                "index": 0,
                "shape": np.array([1, 4], dtype=np.int32),
                "dtype": np.float32,
            }]

        def set_tensor(self, _index, value):
            self._input = value.astype(np.float32)

        def invoke(self):
            pass

        def get_tensor(self, _index):
            value = float(np.mean(self._input))
            return np.array([[value, value * 0.5, value * 0.25, 1.0]], dtype=np.float32)

    def test_waits_for_full_window_then_returns_finite_score(self):
        from scoring.scratch_similarity import ScratchPoseSimilarity

        calc = ScratchPoseSimilarity(
            model_path="unused.tflite",
            sequence_length=3,
            feature_dims=2,
            interpreter=self.FakeInterpreter(),
        )
        reference = np.ones((5, 33, 4), dtype=np.float32)
        user = np.ones((33, 4), dtype=np.float32)

        assert calc.compute(user, reference, 0) is None
        assert calc.compute(user, reference, 1) is None
        score = calc.compute(user, reference, 2)

        assert isinstance(score, float)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_resolve_model_name_from_registry(self, tmp_path):
        from scoring.scratch_similarity import resolve_scratch_model_path

        model_dir = tmp_path / "scratch"
        model_dir.mkdir()
        expected = model_dir / "gcn_e64.tflite"
        registry = {
            "variants": [{
                "name": "gcn_e64",
                "paths": {"tflite": str(expected)},
            }]
        }
        (model_dir / "model_registry.json").write_text(
            __import__("json").dumps(registry), encoding="utf-8")

        resolved = resolve_scratch_model_path(
            model_name="gcn_e64",
            model_dir=str(model_dir),
        )
        assert resolved == str(expected)
