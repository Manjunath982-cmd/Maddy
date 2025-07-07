import sqlite3
from typing import List, Dict, Any


class CalorieEstimator:
    """A tiny calorie estimator that looks up calories/100g from SQLite and assumes 100 g per detection."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(self, detections: List[Dict[str, Any]], image_path: str = "") -> Dict[str, Any]:
        """Return per-item and total calorie information.

        Currently each detected item is assumed to weigh 100 g. You can replace
        this heuristic later with something that uses bounding-box area, depth,
        or reference-object scaling.
        """
        items = []
        total_calories = 0.0

        for det in detections:
            label = det.get("label", "unknown").lower()
            calories_per_100g = self._lookup_calories(label)
            calories = calories_per_100g  # 100g assumption
            items.append({
                "label": det.get("label"),
                "confidence": det.get("confidence"),
                "calories": calories
            })
            total_calories += calories

        return {
            "items": items,
            "total_calories": round(total_calories, 1)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _lookup_calories(self, label: str) -> float:
        """Return calories per 100 g for the given label, or a default if not found."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT calories_per_100g FROM foods WHERE lower(name) = ?", (label,))
        row = c.fetchone()
        conn.close()
        if row:
            return row[0]
        # Default for unknown items
        return 200.0