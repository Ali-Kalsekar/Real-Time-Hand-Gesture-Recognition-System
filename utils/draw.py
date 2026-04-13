from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import mediapipe as mp
import numpy as np

from hand_detection.hand_detector import DetectedHand


def draw_hand_annotations(
    frame: np.ndarray,
    hand: DetectedHand,
    gesture_label: str,
    confidence: float,
    dynamic_label: str,
) -> None:
    x1, y1, x2, y2 = hand.bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 20), 2)

    text_y = max(20, y1 - 10)
    cv2.putText(
        frame,
        f"Gesture: {gesture_label}",
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (x1, text_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Hand: {hand.handedness} ({hand.handedness_score:.2f})",
        (x1, text_y + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    if dynamic_label != "NONE":
        cv2.putText(
            frame,
            f"Dynamic: {dynamic_label}",
            (x1, text_y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 240, 240),
            2,
            cv2.LINE_AA,
        )

    _draw_connections(frame, hand.pixel_landmarks)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 255, 30),
        2,
        cv2.LINE_AA,
    )


def draw_collection_status(frame: np.ndarray, label: str, count: int, target: int) -> None:
    cv2.putText(
        frame,
        f"Collecting: {label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Samples: {count}/{target}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_connections(frame: np.ndarray, pixel_landmarks: Iterable[Tuple[int, int]]) -> None:
    points = list(pixel_landmarks)
    if len(points) != 21:
        return

    for connection in mp.solutions.hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        x1, y1 = points[start_idx]
        x2, y2 = points[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 3, (255, 100, 30), -1)
