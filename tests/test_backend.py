import numpy as np

from ort_gait.backends import CpuBackend, single_weights_array
from ort_gait.config import StitchConfig


def test_original_even_weight_profile_is_preserved() -> None:
    weights = single_weights_array(4)
    np.testing.assert_allclose(weights, [0.0, 1.0, 1.0, 0.0])


def test_cpu_backend_applies_translation_homography() -> None:
    config = StitchConfig(
        panorama_size=(12, 8),
        weight_exponent=6.5,
        output_jpeg_quality=80,
        max_output_jpeg_bytes=1_048_576,
    )
    backend = CpuBackend(config)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    homography = np.array(
        [[1.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    backend.add_image(image, homography, np.eye(3, dtype=np.float32))
    panorama = backend.snapshot()

    assert panorama.shape == (8, 12, 3)
    np.testing.assert_array_equal(panorama[4, 6], [10, 20, 30])
    np.testing.assert_array_equal(panorama[0, 0], [0, 0, 0])
