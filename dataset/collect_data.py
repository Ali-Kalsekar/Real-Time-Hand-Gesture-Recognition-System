from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import cv2

from gesture_recognition.gesture_classifier import DEFAULT_GESTURE_LABELS, preprocess_landmarks
from hand_detection.hand_detector import HandDetector
from utils.draw import draw_collection_status


def load_simple_yaml(path: str) -> Dict[str, str]:
    config: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return config

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        config[k.strip()] = v.strip()
    return config


class DatasetCollector:
    def __init__(
        self,
        output_csv: str,
        camera_index: int,
        max_num_hands: int,
        target_samples: int,
    ) -> None:
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        self.capture = cv2.VideoCapture(camera_index)
        self.detector = HandDetector(max_num_hands=max_num_hands)
        self.target_samples = target_samples
        self.sample_count = 0
        self.last_capture_time = 0.0
        self.capture_interval_sec = 0.1

        self._initialize_csv()

    def _initialize_csv(self) -> None:
        if self.output_csv.exists():
            return
        header = ["label"] + [f"f{i}" for i in range(63)]
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def collect(self, initial_label: str) -> None:
        current_label = initial_label
        gesture_keys = {str(i + 1): label for i, label in enumerate(DEFAULT_GESTURE_LABELS)}

        print("Data Collection Mode")
        print("Press number keys to switch label:")
        for k, v in gesture_keys.items():
            print(f"  {k}: {v}")
        print("Press 'q' to stop.")

        while self.capture.isOpened() and self.sample_count < self.target_samples:
            ok, frame = self.capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            hands = self.detector.process(frame)

            if hands and (time.time() - self.last_capture_time) > self.capture_interval_sec:
                features = preprocess_landmarks(hands[0].landmarks)
                self._save_sample(current_label, features)
                self.sample_count += 1
                self.last_capture_time = time.time()

            draw_collection_status(frame, current_label, self.sample_count, self.target_samples)
            cv2.imshow("Real-Time Hand Gesture Recognition System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            key_chr = chr(key) if 0 <= key <= 255 else ""
            if key_chr in gesture_keys:
                current_label = gesture_keys[key_chr]
                print(f"Switched label to: {current_label}")

        self.close()
        print(f"Saved {self.sample_count} samples to {self.output_csv}")

    def _save_sample(self, label: str, features) -> None:
        with self.output_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([label] + features.tolist())

    def close(self) -> None:
        self.capture.release()
        self.detector.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hand gesture dataset.")
    parser.add_argument("--label", type=str, default="FIST", help="Initial gesture label.")
    parser.add_argument("--samples", type=int, default=600, help="Target number of samples.")
    parser.add_argument("--output", type=str, default="dataset/gesture_data.csv", help="Output CSV path.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    camera_index = int(cfg.get("camera_index", "0"))
    max_num_hands = int(cfg.get("max_num_hands", "2"))

    collector = DatasetCollector(
        output_csv=args.output,
        camera_index=camera_index,
        max_num_hands=max_num_hands,
        target_samples=args.samples,
    )
    collector.collect(args.label.upper())


if __name__ == "__main__":
    main()
