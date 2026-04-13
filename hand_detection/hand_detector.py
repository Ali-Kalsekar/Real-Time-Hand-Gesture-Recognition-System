from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class DetectedHand:
    landmarks: np.ndarray
    pixel_landmarks: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]
    handedness: str
    handedness_score: float


class HandDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[DetectedHand]:
        height, width = frame_bgr.shape[:2]
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        detected_hands: List[DetectedHand] = []

        if not results.multi_hand_landmarks:
            return detected_hands

        handedness_list = results.multi_handedness or []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            normalized_landmarks = []
            pixel_landmarks: List[Tuple[int, int]] = []

            for lm in hand_landmarks.landmark:
                x_norm, y_norm, z_norm = lm.x, lm.y, lm.z
                normalized_landmarks.append([x_norm, y_norm, z_norm])

                x_px = min(max(int(x_norm * width), 0), width - 1)
                y_px = min(max(int(y_norm * height), 0), height - 1)
                pixel_landmarks.append((x_px, y_px))

            lm_array = np.asarray(normalized_landmarks, dtype=np.float32)
            xs = [p[0] for p in pixel_landmarks]
            ys = [p[1] for p in pixel_landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            margin = 16
            bbox = (
                max(0, x_min - margin),
                max(0, y_min - margin),
                min(width - 1, x_max + margin),
                min(height - 1, y_max + margin),
            )

            hand_label = "Unknown"
            hand_score = 0.0
            if idx < len(handedness_list):
                hand_info = handedness_list[idx].classification[0]
                hand_label = hand_info.label
                hand_score = float(hand_info.score)

            detected_hands.append(
                DetectedHand(
                    landmarks=lm_array,
                    pixel_landmarks=pixel_landmarks,
                    bbox=bbox,
                    handedness=hand_label,
                    handedness_score=hand_score,
                )
            )

        return detected_hands

    def close(self) -> None:
        self.hands.close()
