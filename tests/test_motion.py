import numpy as np
import pytest

from ort_gait.config import DynamicConfig
from ort_gait.motion import MotionDetector, shannon_entropy


def test_binary_entropy() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 255
    assert shannon_entropy(mask) == pytest.approx(1.0)


def test_static_frame_is_skipped_and_dynamic_frame_passes() -> None:
    detector = MotionDetector(DynamicConfig(30, 0.10))
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    changed = black.copy()
    changed[:, :5] = 255

    assert detector.is_dynamic("cam0", black)[0] is False
    assert detector.is_dynamic("cam0", black)[0] is False
    is_dynamic, entropy = detector.is_dynamic("cam0", changed)
    assert is_dynamic is True
    assert entropy == pytest.approx(1.0)
