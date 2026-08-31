from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from ort_gait.app import IpcApplication, PcApplication
from ort_gait.backends import create_backend
from ort_gait.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ORT-GAIT multi-camera panorama runtime"
    )
    parser.add_argument(
        "--role",
        choices=("ipc", "pc"),
        required=True,
        help="ipc: capture/send cameras; pc: receive/stitch/serve Quest",
    )
    parser.add_argument(
        "--cam_no",
        type=int,
        default=None,
        help="number of cameras to enable, selected from config.yaml in order",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="path to YAML configuration",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="stitching device for the pc role",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the PC debug panorama window",
    )
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    )

    try:
        config = load_config(args.config, cam_no=args.cam_no)
        if args.preview:
            config.runtime.preview = True

        if args.role == "ipc":
            IpcApplication(config).run()
        else:
            backend = create_backend(args.device, config.stitch)
            PcApplication(config, backend).run()
    except (ConfigError, RuntimeError) as exc:
        logging.error("启动失败：%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.info("收到退出信号。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
