import numpy as np
import pytest
import warnings

from ort_gait.backends import CudaBackend
from ort_gait.config import StitchConfig


def _cuda_runtime_ready() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="CUDA path could not be detected.*"
            )
            import cupy as cp

        value = (cp.arange(2, dtype=cp.float32) + 1).sum().get()
        return float(value) == 3.0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_runtime_ready(), reason="CUDA runtime is unavailable")
def test_cuda_backend_executes_warp_and_returns_panorama() -> None:
    config = StitchConfig((12, 8), 6.5, 80, 1_048_576)
    backend = CudaBackend(config)
    image = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
    homography = np.array(
        [[1.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    backend.add_image(image, homography, np.eye(3, dtype=np.float32))
    panorama = backend.snapshot()
    np.testing.assert_array_equal(panorama[4, 6], [10, 20, 30])
