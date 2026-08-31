from pathlib import Path

import pytest

from ort_gait.config import load_config
from ort_gait.geometry import HeadYawTracker, determine_visible_cameras


CONFIG = Path(__file__).parents[1] / "config.yaml"


def test_head_yaw_is_relative_to_first_sample_and_base_minus_30() -> None:
    tracker = HeadYawTracker(-30.0)
    assert tracker.update(0.0) == pytest.approx(-30.0)
    assert tracker.update(355.0) == pytest.approx(-35.0)
    # Moving +10 degrees from 355 reaches 5 degrees.
    assert tracker.update(5.0) == pytest.approx(-25.0)


def test_visibility_wraps_around_minus_180_to_180() -> None:
    config = load_config(CONFIG)
    visible = determine_visible_cameras(-170.0, 80.0, config.cameras)
    assert "cam0" in visible
    assert "cam5" in visible
