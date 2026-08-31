from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ort_gait.config import CameraConfig


def normalize_degrees(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


@dataclass
class HeadYawTracker:
    """Convert Unity absolute yaw to a delta around a configured base yaw."""

    base_yaw_deg: float
    reference_yaw_deg: float | None = None
    current_yaw_deg: float | None = None

    def __post_init__(self) -> None:
        self.current_yaw_deg = normalize_degrees(self.base_yaw_deg)

    def update(self, unity_yaw_deg: float) -> float:
        if not isfinite(unity_yaw_deg):
            raise ValueError("yaw must be finite")
        raw = normalize_degrees(unity_yaw_deg)
        if self.reference_yaw_deg is None:
            self.reference_yaw_deg = raw
        relative = normalize_degrees(raw - self.reference_yaw_deg)
        self.current_yaw_deg = normalize_degrees(self.base_yaw_deg + relative)
        return self.current_yaw_deg


def _circular_intervals(start_deg: float, end_deg: float) -> list[tuple[float, float]]:
    start = start_deg % 360.0
    end = end_deg % 360.0
    if start <= end:
        return [(start, end)]
    return [(start, 360.0), (0.0, end)]


def determine_visible_cameras(
    yaw_deg: float,
    fov_deg: float,
    cameras: Iterable[CameraConfig],
) -> list[str]:
    half_fov = fov_deg / 2.0
    view_intervals = _circular_intervals(yaw_deg - half_fov, yaw_deg + half_fov)
    visible: list[str] = []
    for camera in cameras:
        camera_intervals = _circular_intervals(*camera.angle_range_deg)
        if any(
            max(view_start, cam_start) < min(view_end, cam_end)
            for view_start, view_end in view_intervals
            for cam_start, cam_end in camera_intervals
        ):
            visible.append(camera.name)
    return visible
