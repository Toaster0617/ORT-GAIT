from __future__ import annotations

import cv2
import numpy as np

from ort_gait.config import DynamicConfig


def shannon_entropy(image: np.ndarray) -> float:
    counts = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / image.size
    return float(-(probabilities * np.log2(probabilities)).sum())


class MotionDetector:
    """The original frame-difference + binary Shannon entropy gate."""

    def __init__(self, config: DynamicConfig) -> None:
        self._config = config
        self._previous: dict[str, np.ndarray] = {}

    def is_dynamic(self, camera_name: str, image: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        previous = self._previous.get(camera_name)
        self._previous[camera_name] = gray
        if previous is None:
            return False, 0.0

        difference = cv2.absdiff(previous, gray)
        _, mask = cv2.threshold(
            difference,
            self._config.pixel_difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        entropy = shannon_entropy(mask)
        return entropy >= self._config.entropy_threshold, entropy
