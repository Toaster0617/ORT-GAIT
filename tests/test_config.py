from pathlib import Path

import pytest

from ort_gait.config import ConfigError, load_config


CONFIG = Path(__file__).parents[1] / "config.yaml"


def test_cam_no_selects_first_two_cameras() -> None:
    config = load_config(CONFIG, cam_no=2)
    assert [camera.name for camera in config.cameras] == ["cam0", "cam1"]


def test_four_ports_are_distinct_and_match_quest_demo() -> None:
    network = load_config(CONFIG).network
    ports = {
        network.camera_udp_port,
        network.visibility_udp_port,
        network.quest_image_tcp_port,
        network.quest_yaw_udp_port,
    }
    assert ports == {7000, 8001, 8082, 8084}


def test_invalid_cam_no_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(CONFIG, cam_no=0)
