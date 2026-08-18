#!/usr/bin/env python3
"""Verify the Prompt 0 container without parsing any TFRecord record."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def count_tfrecords(root: Path) -> int:
    return sum(1 for p in root.rglob("*.tfrecord*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--womd-root", default=os.environ.get("WOMD_ROOT", "/mnt/womd"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "womd_root": args.womd_root,
        "record_parse_performed": False,
        "imports": {},
        "splits": {},
    }

    import_errors: list[str] = []
    try:
        import numpy as np

        result["imports"]["numpy"] = np.__version__
    except Exception as exc:  # pragma: no cover - diagnostic path
        import_errors.append(f"numpy: {exc}")
    try:
        import pandas as pd

        result["imports"]["pandas"] = pd.__version__
    except Exception as exc:
        import_errors.append(f"pandas: {exc}")
    try:
        import pyarrow as pa

        result["imports"]["pyarrow"] = pa.__version__
    except Exception as exc:
        import_errors.append(f"pyarrow: {exc}")
    try:
        import tensorflow as tf

        result["imports"]["tensorflow"] = tf.__version__
    except Exception as exc:
        import_errors.append(f"tensorflow: {exc}")
    try:
        import google.protobuf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.protos import scenario_pb2

        # Instantiate messages only. Do not parse a dataset record in this step.
        dataset_pb2.Frame()
        scenario_pb2.Scenario()
        result["imports"]["protobuf"] = getattr(google.protobuf, "__version__", "unknown")
        result["imports"]["waymo_dataset_pb2"] = "ok"
        result["imports"]["waymo_scenario_pb2"] = "ok"
    except Exception as exc:
        import_errors.append(f"waymo protobuf modules: {exc}")

    womd_root = Path(args.womd_root)
    if not womd_root.is_dir():
        import_errors.append(f"WOMD root is not a directory: {womd_root}")
    else:
        for split in ("training", "validation", "testing"):
            split_root = womd_root / split
            result["splits"][split] = {
                "exists": split_root.is_dir(),
                "readable": os.access(split_root, os.R_OK),
                "tfrecord_file_count": count_tfrecords(split_root) if split_root.is_dir() else 0,
            }
            if not split_root.is_dir():
                import_errors.append(f"missing split: {split_root}")
            elif result["splits"][split]["tfrecord_file_count"] == 0:
                import_errors.append(f"no TFRecord files under: {split_root}")

    result["errors"] = import_errors
    result["status"] = "PASS" if not import_errors else "FAIL"

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0 if not import_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
