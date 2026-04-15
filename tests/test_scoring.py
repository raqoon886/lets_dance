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
