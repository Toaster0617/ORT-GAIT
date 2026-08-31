import numpy as np

from ort_gait.state import ReceiverState, SenderState


def test_sender_starts_with_all_cameras_visible() -> None:
    state = SenderState({"cam0", "cam1"})
    assert state.visible_snapshot() == {"cam0", "cam1"}


def test_receiver_waits_for_all_numpy_images_without_truth_value_error() -> None:
    state = ReceiverState(("cam0", "cam1"), -30.0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state.update_image("cam0", image, "10.0.0.2")
    assert state.wait_for_all_images(0.0) is None
    state.update_image("cam1", image, "10.0.0.2")
    images = state.wait_for_all_images(0.0)
    assert images is not None
    assert set(images) == {"cam0", "cam1"}


def test_latest_jpeg_is_shared_between_stitcher_and_tcp_server() -> None:
    state = ReceiverState(("cam0",), -30.0)
    state.publish_jpeg(b"jpeg")
    version, jpeg = state.wait_for_jpeg(0, 0.0)
    assert version == 1
    assert jpeg == b"jpeg"
