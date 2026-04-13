from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import cv2
import tensorflow as tf

from gesture_recognition.gesture_classifier import GestureClassifier
from hand_detection.hand_detector import HandDetector
from utils.draw import draw_fps, draw_hand_annotations
from utils.fps import FPSCounter
from utils.logger import PredictionLogger


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


def parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def majority_vote(history: Deque[str]) -> str:
    if not history:
        return "UNKNOWN"
    return Counter(history).most_common(1)[0][0]


def detect_dynamic_gesture(points: Deque[Tuple[int, int]]) -> str:
    if len(points) < 4:
        return "NONE"

    x0, y0 = points[0]
    x1, y1 = points[-1]
    dx, dy = x1 - x0, y1 - y0

    threshold = 40
    if abs(dx) > abs(dy) and abs(dx) > threshold:
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    if abs(dy) > abs(dx) and abs(dy) > threshold:
        return "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"
    return "NONE"


def configure_gpu_if_available(enabled: bool) -> None:
    if not enabled:
        return

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time Hand Gesture Recognition System")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, default="", help="Video path. If empty, webcam is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)

    camera_index = int(cfg.get("camera_index", "0"))
    confidence_threshold = float(cfg.get("confidence_threshold", "0.6"))
    max_num_hands = int(cfg.get("max_num_hands", "2"))
    model_path = cfg.get("model_path", "models/gesture_model.h5")
    labels_path = cfg.get("labels_path", "models/gesture_labels.json")
    log_path = cfg.get("log_path", "output/predictions_log.csv")
    frame_width = int(cfg.get("frame_width", "960"))
    frame_height = int(cfg.get("frame_height", "540"))
    smoothing_window = int(cfg.get("prediction_smoothing_window", "5"))
    dynamic_window = int(cfg.get("dynamic_window", "8"))
    use_gpu = parse_bool(cfg.get("use_gpu_if_available", "true"), default=True)

    configure_gpu_if_available(use_gpu)

    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Run training first: python train_model.py")
        return

    detector = HandDetector(max_num_hands=max_num_hands)
    classifier = GestureClassifier(
        model_path=model_path,
        labels_path=labels_path,
        confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window,
    )

    history_by_hand = defaultdict(lambda: deque(maxlen=smoothing_window))
    motion_by_hand = defaultdict(lambda: deque(maxlen=dynamic_window))

    logger = PredictionLogger(log_path)
    fps_counter = FPSCounter()

    source = args.video if args.video else camera_index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    window_title = "Real-Time Hand Gesture Recognition System"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if not args.video:
            frame = cv2.flip(frame, 1)

        hands = detector.process(frame)
        fps = fps_counter.update()

        for idx, hand in enumerate(hands):
            label, confidence, _ = classifier.predict_from_landmarks(hand.landmarks)

            history_by_hand[idx].append(label)
            smoothed_label = majority_vote(history_by_hand[idx])

            wrist_xy = hand.pixel_landmarks[0]
            motion_by_hand[idx].append(wrist_xy)
            dynamic_label = detect_dynamic_gesture(motion_by_hand[idx])

            draw_hand_annotations(frame, hand, smoothed_label, confidence, dynamic_label)

            if frame_id % 5 == 0:
                logger.log(smoothed_label, confidence, hand.handedness, dynamic_label, fps)

        draw_fps(frame, fps)
        cv2.imshow(window_title, frame)

        frame_id += 1
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
