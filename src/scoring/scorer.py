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
                 grade_thresholds=None, baseline=0.0):
        self.score_scale = score_scale
        self.combo_multiplier = combo_multiplier
        # baseline 보정: raw similarity에서 baseline을 빼고 유효 범위를 0~1로 리매핑
        # 예) baseline=0.65이면 raw 0.65→0, raw 1.0→1.0, raw 0.82→0.49
        self.baseline = float(baseline)
        self.grade_thresholds = grade_thresholds or {
            "perfect": 90,
            "great": 75,
            "good": 60,
            "ok": 40,
        }
        # 콤보 티어: (최소콤보, 배율)
        self.combo_tiers = [
            (40, 1.5),
            (20, 1.3),
            (10, 1.2),
            (5,  1.1),
        ]
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
        # baseline 보정 후 0~100 스케일로 변환
        if self.baseline > 0:
            adjusted = (similarity - self.baseline) / (1.0 - self.baseline)
            adjusted = max(0.0, min(1.0, adjusted))
        else:
            adjusted = max(0.0, min(1.0, similarity))
        score_100 = adjusted * self.score_scale
        grade = self._get_grade(score_100)

        # 콤보: Perfect만 증가, Miss만 리셋, 나머지(Great/Good/OK)는 유지
        if grade == "Perfect":
            self._combo_count += 1
        elif grade == "Miss":
            self._combo_count = 0
        # Great / Good / OK: combo 유지 (변경 없음)

        self._max_combo = max(self._max_combo, self._combo_count)

        # 티어 기반 콤보 보너스
        combo_bonus = 1.0
        for tier_min, tier_mult in self.combo_tiers:
            if self._combo_count >= tier_min:
                combo_bonus = tier_mult
                break
        points = score_100 * combo_bonus
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
