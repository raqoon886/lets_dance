"""
Dance Scorer - Converts raw similarity scores into game scores with combos.
"""


class DanceScorer:
    """
    Manages scoring logic: converts similarity to points, tracks combos,
    and computes final grades.
    """

    GRADES = ["Perfect", "Great", "Good", "OK", "Miss"]

    def __init__(self, score_scale=100, combo_multiplier=1.5,
                 grade_thresholds=None):
        self.score_scale = score_scale
        self.combo_multiplier = combo_multiplier
        self.grade_thresholds = grade_thresholds or {
            "perfect": 90,
            "great": 75,
            "good": 60,
            "ok": 40,
        }
        self._total_score = 0
        self._combo_count = 0
        self._max_combo = 0
        self._hit_counts = {grade: 0 for grade in self.GRADES}
        self._history = []

    def reset(self):
        """Reset all score tracking for a new game session."""
        self._total_score = 0
        self._combo_count = 0
        self._max_combo = 0
        self._hit_counts = {grade: 0 for grade in self.GRADES}
        self._history.clear()

    def evaluate(self, similarity: float) -> dict:
        """
        Evaluate a single similarity score and update game state.

        Args:
            similarity: Raw similarity score (0.0 to 1.0)

        Returns:
            Dict with 'grade', 'points', 'combo', 'total_score'
        """
        # TODO:
        # 1. Map similarity to grade
        # 2. Calculate base points
        # 3. Apply combo multiplier
        # 4. Update combo streak
        # 5. Track hit counts
        score_100 = similarity * self.score_scale
        grade = self._get_grade(score_100)

        if grade != "Miss":
            self._combo_count += 1
        else:
            self._combo_count = 0

        self._max_combo = max(self._max_combo, self._combo_count)

        combo_bonus = 1.0 + (self._combo_count * 0.1)
        points = score_100 * min(combo_bonus, self.combo_multiplier)
        self._total_score += points
        self._hit_counts[grade] += 1
        self._history.append(score_100)

        return {
            "grade": grade,
            "points": round(points, 1),
            "combo": self._combo_count,
            "total_score": round(self._total_score, 1),
        }

    def get_final_result(self) -> dict:
        """
        Get final game result summary.

        Returns:
            Dict with final score, grade distribution, max combo, average score
        """
        avg = sum(self._history) / len(self._history) if self._history else 0
        return {
            "total_score": round(self._total_score, 1),
            "average_score": round(avg, 1),
            "max_combo": self._max_combo,
            "hit_counts": dict(self._hit_counts),
            "total_moves": len(self._history),
            "final_grade": self._get_grade(avg),
        }

    def _get_grade(self, score: float) -> str:
        """Map a 0-100 score to a grade string."""
        if score >= self.grade_thresholds["perfect"]:
            return "Perfect"
        elif score >= self.grade_thresholds["great"]:
            return "Great"
        elif score >= self.grade_thresholds["good"]:
            return "Good"
        elif score >= self.grade_thresholds["ok"]:
            return "OK"
        return "Miss"

    @property
    def total_score(self) -> float:
        return self._total_score

    @property
    def combo(self) -> int:
        return self._combo_count
