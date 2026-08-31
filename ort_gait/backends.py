from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import sys
from typing import Any
import warnings

import cv2
import numpy as np

from ort_gait.config import StitchConfig


LOGGER = logging.getLogger(__name__)


def single_weights_array(size: int) -> np.ndarray:
    if size % 2 == 1:
        return np.concatenate(
            (
                np.linspace(0.0, 1.0, (size + 1) // 2),
                np.linspace(1.0, 0.0, (size + 1) // 2)[1:],
            )
        )
    return np.concatenate(
        (
            np.linspace(0.0, 1.0, size // 2),
            np.linspace(1.0, 0.0, size // 2),
        )
    )


class WeightFactory:
    def __init__(self, exponent: float) -> None:
        self._exponent = exponent
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def get(self, shape: tuple[int, int]) -> np.ndarray:
        if shape not in self._cache:
            height, width = shape
            row = single_weights_array(height)[:, None]
            column = single_weights_array(width)[None, :]
            weight = (row @ column) ** self._exponent
            self._cache[shape] = np.repeat(
                weight[:, :, None].astype(np.float32), 3, axis=2
            )
        return self._cache[shape]


def _roi_for(
    image_shape: tuple[int, int],
    homography: np.ndarray,
    panorama_size: tuple[int, int],
) -> tuple[int, int, int, int, np.ndarray] | None:
    height, width = image_shape
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=np.float32,
    ).T
    warped = homography @ corners
    warped /= warped[2]
    xs, ys = warped[0], warped[1]
    xmin = max(int(xs.min()) - 1, 0)
    ymin = max(int(ys.min()) - 1, 0)
    xmax = min(int(xs.max()) + 1, panorama_size[0])
    ymax = min(int(ys.max()) + 1, panorama_size[1])
    if xmax <= xmin or ymax <= ymin:
        return None
    translation = np.array(
        [[1.0, 0.0, -xmin], [0.0, 1.0, -ymin], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return xmin, ymin, xmax, ymax, translation @ homography


class PanoramaBackend(ABC):
    @abstractmethod
    def add_image(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def snapshot(self) -> np.ndarray:
        pass


class CpuBackend(PanoramaBackend):
    """NumPy/OpenCV implementation of the original warp-and-feather math."""

    def __init__(self, config: StitchConfig) -> None:
        self._size = config.panorama_size
        width, height = self._size
        self._panorama = np.zeros((height, width, 3), dtype=np.uint8)
        self._weights = np.zeros((height, width, 3), dtype=np.float32)
        self._weight_factory = WeightFactory(config.weight_exponent)
        self._has_image = False

    def add_image(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        if self._has_image and not np.array_equal(offset, np.eye(3, dtype=np.float32)):
            self._panorama = cv2.warpPerspective(self._panorama, offset, self._size)
            self._weights = cv2.warpPerspective(self._weights, offset, self._size)

        roi = _roi_for(image.shape[:2], homography, self._size)
        if roi is None:
            return
        xmin, ymin, xmax, ymax, local_h = roi
        roi_size = (xmax - xmin, ymax - ymin)
        new_image = cv2.warpPerspective(image, local_h, roi_size)
        new_weight = cv2.warpPerspective(
            self._weight_factory.get(image.shape[:2]), local_h, roi_size
        )

        old_image = self._panorama[ymin:ymax, xmin:xmax]
        old_weight = self._weights[ymin:ymax, xmin:xmax]
        normalized_old = old_weight[:, :, 0] / (
            old_weight[:, :, 0] + new_weight[:, :, 0] + 1e-6
        )
        normalized_old = normalized_old[:, :, None]
        blended = new_image * (1.0 - normalized_old) + old_image * normalized_old
        self._panorama[ymin:ymax, xmin:xmax] = np.clip(blended, 0, 255).astype(
            np.uint8
        )

        combined = old_weight + new_weight
        maximum = float(combined.max())
        if maximum != 0.0:
            combined = combined / maximum
        self._weights[ymin:ymax, xmin:xmax] = combined.astype(np.float32)
        self._has_image = True

    def snapshot(self) -> np.ndarray:
        return self._panorama.copy()


class CudaBackend(PanoramaBackend):
    """CuPy CUDA backend; image tensors remain on the GPU between additions."""

    def __init__(self, config: StitchConfig) -> None:
        import cupy as cp
        from cupyx.scipy.ndimage import map_coordinates

        self._cp = cp
        self._map_coordinates = map_coordinates
        self._size = config.panorama_size
        width, height = self._size
        self._panorama = cp.zeros((height, width, 3), dtype=cp.uint8)
        self._weights = cp.zeros((height, width, 3), dtype=cp.float32)
        self._weight_factory = WeightFactory(config.weight_exponent)
        self._gpu_weights: dict[tuple[int, int], Any] = {}
        self._map_cache: dict[tuple[bytes, int, int], Any] = {}
        self._has_image = False

    def _coordinates(
        self,
        homography: np.ndarray,
        output_size: tuple[int, int],
    ) -> Any:
        width, height = output_size
        matrix = np.asarray(homography, dtype=np.float32)
        key = (matrix.tobytes(), width, height)
        cached = self._map_cache.get(key)
        if cached is not None:
            return cached

        cp = self._cp
        inverse = cp.asarray(np.linalg.inv(matrix), dtype=cp.float32)
        output_y, output_x = cp.indices((height, width), dtype=cp.float32)
        denominator = (
            inverse[2, 0] * output_x
            + inverse[2, 1] * output_y
            + inverse[2, 2]
        )
        source_x = (
            inverse[0, 0] * output_x
            + inverse[0, 1] * output_y
            + inverse[0, 2]
        ) / denominator
        source_y = (
            inverse[1, 0] * output_x
            + inverse[1, 1] * output_y
            + inverse[1, 2]
        ) / denominator
        coordinates = cp.stack((source_y, source_x))
        self._map_cache[key] = coordinates
        return coordinates

    def _warp(self, source: Any, homography: np.ndarray, output_size: tuple[int, int]) -> Any:
        cp = self._cp
        coordinates = self._coordinates(homography, output_size)
        source_float = source.astype(cp.float32, copy=False)
        channels = [
            self._map_coordinates(
                source_float[:, :, channel],
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for channel in range(source.shape[2])
        ]
        warped = cp.stack(channels, axis=2)
        if source.dtype == cp.uint8:
            return cp.rint(cp.clip(warped, 0, 255)).astype(cp.uint8)
        return warped.astype(source.dtype, copy=False)

    def _weight(self, shape: tuple[int, int]) -> Any:
        if shape not in self._gpu_weights:
            self._gpu_weights[shape] = self._cp.asarray(self._weight_factory.get(shape))
        return self._gpu_weights[shape]

    def add_image(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        cp = self._cp
        if self._has_image and not np.array_equal(offset, np.eye(3, dtype=np.float32)):
            self._panorama = self._warp(self._panorama, offset, self._size)
            self._weights = self._warp(self._weights, offset, self._size)

        roi = _roi_for(image.shape[:2], homography, self._size)
        if roi is None:
            return
        xmin, ymin, xmax, ymax, local_h = roi
        roi_size = (xmax - xmin, ymax - ymin)
        # One host-to-device upload for the newly received camera frame.
        gpu_image = cp.asarray(image)
        new_image = self._warp(gpu_image, local_h, roi_size)
        new_weight = self._warp(self._weight(image.shape[:2]), local_h, roi_size)

        old_image = self._panorama[ymin:ymax, xmin:xmax]
        old_weight = self._weights[ymin:ymax, xmin:xmax]
        normalized_old = old_weight[:, :, 0] / (
            old_weight[:, :, 0] + new_weight[:, :, 0] + 1e-6
        )
        normalized_old = normalized_old[:, :, None]
        blended = new_image * (1.0 - normalized_old) + old_image * normalized_old
        self._panorama[ymin:ymax, xmin:xmax] = cp.clip(blended, 0, 255).astype(
            cp.uint8
        )

        combined = old_weight + new_weight
        maximum = combined.max()
        combined = cp.where(maximum != 0.0, combined / maximum, combined)
        self._weights[ymin:ymax, xmin:xmax] = combined.astype(cp.float32)
        self._has_image = True

    def snapshot(self) -> np.ndarray:
        # The only full-frame device-to-host transfer in the CUDA pipeline.
        return self._cp.asnumpy(self._panorama)


def _cuda_available() -> tuple[bool, str]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="CUDA path could not be detected.*"
            )
            import cupy as cp
            from cupyx.scipy.ndimage import map_coordinates

        if cp.cuda.runtime.getDeviceCount() < 1:
            return False, "未检测到 CUDA 设备"
        probe = cp.arange(4, dtype=cp.float32).reshape(2, 2)
        coordinates = cp.indices((2, 2), dtype=cp.float32)
        result = map_coordinates(
            probe, coordinates, order=1, mode="constant", prefilter=False
        )
        if float(result.sum().get()) != 6.0:
            return False, "CUDA 计算探针结果异常"
        cp.cuda.get_current_stream().synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def create_backend(device: str, config: StitchConfig) -> PanoramaBackend:
    if device == "cpu":
        _announce_device("CPU")
        return CpuBackend(config)

    available, reason = _cuda_available()
    if available:
        _announce_device("GPU")
        return CudaBackend(config)
    if device == "gpu":
        raise RuntimeError(f"请求了 GPU，但 CUDA 不可用：{reason}")
    _announce_device("CPU")
    LOGGER.info("CUDA 不可用，自动使用 CPU：%s", reason)
    return CpuBackend(config)


def _announce_device(device: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"当前使用：{device}")
