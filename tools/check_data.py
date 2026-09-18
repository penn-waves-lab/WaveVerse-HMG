"""Check required assets, splits and all seven cache files per training motion."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import CACHE_SUFFIXES, DEFAULT_DATA_ROOT, DEFAULT_ASSET_ROOT, require_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--token-root", type=Path)
    args = parser.parse_args()
    token_root = args.token_root or args.data_root / "VQVAE"
    require_assets(args.asset_root)
    for path in ["train.txt", "test.txt", "new_joint_vecs", "texts"]:
        if not (args.data_root / path).exists():
            raise FileNotFoundError(args.data_root / path)
    names = (args.data_root / "train.txt").read_text().splitlines()
    complete = [
        name
        for name in names
        if (args.data_root / "texts" / (name + ".txt")).is_file()
        and all((token_root / (name + suffix)).is_file() for suffix in CACHE_SUFFIXES)
    ]
    print(
        json.dumps(
            {
                "train_split_entries": len(names),
                "complete_cache_entries": len(complete),
                "missing_or_filtered_entries": len(names) - len(complete),
            },
            indent=2,
        )
    )
    if not complete:
        raise RuntimeError("No complete training caches; run prepare_tokens.py first.")


if __name__ == "__main__":
    main()
