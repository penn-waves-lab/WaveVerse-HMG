"""Launch the validated path-conditioned transformer training recipe."""

import argparse
import json
from pathlib import Path
import runpy
import sys

from tools.common import (
    CODE,
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_ASSET_ROOT,
    DEFAULT_TRAIN_OUTPUT,
    configure,
    read_config,
    require_assets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--token-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without loading data or models"
    )
    args, overrides = parser.parse_known_args()
    config = read_config(args.config)
    output = args.output.resolve()
    vq_path = args.asset_root.resolve() / "pretrained/VQVAE/net_last.pth"
    command = ["train_t2m_trans.py"]
    for key, value in config.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command += [option] + [str(v) for v in (value if isinstance(value, list) else [value])]
    command += [
        "--resume-pth",
        str(vq_path),
        "--out-dir",
        str(output.parent),
        "--exp-name",
        output.name,
    ] + overrides
    if args.dry_run:
        print(json.dumps({"cwd": str(CODE), "argv": command}, indent=2))
        return
    # Refuse accidental reuse of an experiment's output directory.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new --output directory: " + str(output))
    _, assets, _ = configure(args.data_root, args.asset_root, args.token_root)
    require_assets(assets)
    sys.argv = command
    runpy.run_path(str(CODE / "train_t2m_trans.py"), run_name="__main__")


if __name__ == "__main__":
    main()
