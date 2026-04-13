from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class PredictionLogger:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        if self.csv_path.exists():
            return

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "gesture",
                    "confidence",
                    "handedness",
                    "dynamic_gesture",
                    "fps",
                ]
            )

    def log(
        self,
        gesture: str,
        confidence: float,
        handedness: str,
        dynamic_gesture: str,
        fps: float,
    ) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="milliseconds"),
                    gesture,
                    round(confidence, 4),
                    handedness,
                    dynamic_gesture,
                    round(fps, 2),
                ]
            )
