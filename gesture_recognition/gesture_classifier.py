from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

DEFAULT_GESTURE_LABELS = ["FIST", "PALM", "THUMBS_UP", "PEACE", "OK", "STOP"]


def preprocess_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.shape != (21, 3):
        raise ValueError("Expected landmarks shape to be (21, 3).")

    base = landmarks[0, :2].copy()
    rel_xy = landmarks[:, :2] - base
    rel_z = landmarks[:, 2:3] - landmarks[0, 2:3]

    max_norm = np.linalg.norm(rel_xy, axis=1).max()
    scale = max(max_norm, 1e-6)

    rel_xy = rel_xy / scale
    rel_z = rel_z / scale

    features = np.hstack([rel_xy, rel_z]).reshape(-1).astype(np.float32)
    return features


class GestureClassifier:
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        smoothing_window: int = 5,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = self._load_labels(labels_path)
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(self.model_path)
        self.prob_history: deque[np.ndarray] = deque(maxlen=max(1, smoothing_window))

    def _load_model(self, model_path: Path) -> tf.keras.Model:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        return tf.keras.models.load_model(model_path)

    def _load_labels(self, labels_path: Optional[str]) -> List[str]:
        if labels_path:
            lp = Path(labels_path)
            if lp.exists():
                with lp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                labels = data.get("labels", [])
                if labels:
                    return labels
        return DEFAULT_GESTURE_LABELS

    def predict_from_landmarks(self, landmarks: np.ndarray) -> Tuple[str, float, np.ndarray]:
        features = preprocess_landmarks(landmarks)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        if features.ndim != 1:
            raise ValueError("Features must be a flattened 1D vector.")

        logits = self.model.predict(features[np.newaxis, ...], verbose=0)[0]
        probs = tf.nn.softmax(logits).numpy() if logits.ndim == 1 else logits

        self.prob_history.append(probs)
        smoothed_probs = np.mean(np.vstack(self.prob_history), axis=0)

        pred_idx = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[pred_idx])

        if pred_idx >= len(self.labels):
            return "UNKNOWN", confidence, smoothed_probs

        label = self.labels[pred_idx]
        if confidence < self.confidence_threshold:
            label = "UNKNOWN"

        return label, confidence, smoothed_probs
