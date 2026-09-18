"""Prepare frozen-VQ tokens, reconstructed features and prefix-state caches."""

import argparse
import json
from pathlib import Path

from tools.common import (
    CACHE_SUFFIXES,
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_ASSET_ROOT,
    configure,
    load_vq,
    read_config,
    require_assets,
    seed_all,
    sha256,
)


def reconstruct_variant(net, codes, mean, std):
    """Decode each prefix: full-sequence positions are not prefix states."""
    import torch
    from utils.motion_process import recover_from_ric

    reconstructed = net.forward_decoder(codes).cpu() * std + mean
    positions = []
    for index in range(codes.shape[1]):
        prefix = net.forward_decoder(codes[:, : index + 1]).cpu() * std + mean
        joints = recover_from_ric(prefix.float(), 22)
        positions.append(joints[0, -1:, 0, [0, 2]])
    return reconstructed.numpy(), torch.cat(positions, dim=0)[None].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--token-root", type=Path, help="New output directory (default: <data-root>/VQVAE)"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Diagnostic limit on motions")
    args = parser.parse_args()
    config = read_config(args.config)
    data, assets, tokens = configure(args.data_root, args.asset_root, args.token_root)
    paths = require_assets(assets, preparation_only=True)
    if tokens.exists() and any(tokens.iterdir()):
        raise FileExistsError(
            "Use a new token directory; existing caches are preserved: " + str(tokens)
        )
    tokens.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from dataset import dataset_tokenize
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("Cache preparation requires CUDA.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    seed_all(args.seed)
    loader, dataset = dataset_tokenize.DATALoader(
        "t2m", 1, num_workers=args.num_workers, unit_length=2 ** config["down_t"]
    )
    if not len(dataset):
        raise RuntimeError("No valid training motions found.")
    net = load_vq(config, paths["vq_checkpoint"])
    mean = torch.from_numpy(dataset.mean)
    std = torch.from_numpy(dataset.std)
    metadata = {
        "seed": args.seed,
        "num_workers": args.num_workers,
        "torch": torch.__version__,
        "config": config,
        "vq_sha256": sha256(paths["vq_checkpoint"]),
        "mean_sha256": sha256(paths["mean"]),
        "std_sha256": sha256(paths["std"]),
        "train_split_sha256": sha256(data / "train.txt"),
        "completed_names": [],
    }
    with torch.no_grad():
        for pose, names in tqdm(loader, desc="Preparing token caches"):
            name = names[0]
            codes = net.encode(pose.cuda().float())
            assert codes.shape[1] > 1
            arrays = {".npy": codes.cpu().numpy()}
            for suffix, variant in [
                ("", codes),
                ("_dropfirstcode", codes[:, 1:]),
                ("_droplastcode", codes[:, :-1]),
            ]:
                features, positions = reconstruct_variant(net, variant, mean, std)
                arrays["_recfeat" + suffix + ".npy"] = features
                arrays["_recincrepos" + suffix + ".npy"] = positions
            assert set(arrays) == set(CACHE_SUFFIXES)
            assert all(np.isfinite(array).all() for array in arrays.values())
            for suffix, array in arrays.items():
                with (tokens / (name + suffix)).open("xb") as stream:
                    np.save(stream, array)
            metadata["completed_names"].append(name)
            (tokens / "preparation.json").write_text(json.dumps(metadata, indent=2))
            if args.limit and len(metadata["completed_names"]) >= args.limit:
                break
    print("Prepared", len(metadata["completed_names"]), "motions in", tokens)


if __name__ == "__main__":
    main()
