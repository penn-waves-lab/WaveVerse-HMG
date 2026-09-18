"""Portable paths and model loading for the training release."""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "waveverse_hmg"
DEFAULT_CONFIG = ROOT / "configs/pathhmg.json"
DEFAULT_DATA_ROOT = ROOT / "dataset/HumanML3D"
DEFAULT_ASSET_ROOT = ROOT / "assets"
DEFAULT_TRAIN_OUTPUT = ROOT / "outputs/waveverse-hmg"
DEFAULT_EVAL_OUTPUT = ROOT / "outputs/waveverse-hmg-evaluation"
CACHE_SUFFIXES = (
    ".npy",
    "_recfeat.npy",
    "_recincrepos.npy",
    "_recfeat_dropfirstcode.npy",
    "_recincrepos_dropfirstcode.npy",
    "_recfeat_droplastcode.npy",
    "_recincrepos_droplastcode.npy",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config["dataname"] != "t2m":
        raise ValueError("This release validates HumanML3D (dataname=t2m).")
    return config


def configure(data_root, asset_root, token_root=None):
    data_root = Path(data_root).resolve()
    asset_root = Path(asset_root).resolve()
    token_root = Path(token_root).resolve() if token_root else data_root / "VQVAE"
    os.environ["HMG_DATA_ROOT"] = str(data_root)
    os.environ["HMG_ASSET_ROOT"] = str(asset_root)
    os.environ["HMG_TOKEN_ROOT"] = str(token_root)
    os.environ.setdefault("WANDB_MODE", "offline")
    sys.path.insert(0, str(CODE))
    os.chdir(CODE)
    return data_root, asset_root, token_root


def asset_paths(asset_root):
    root = Path(asset_root)
    meta = root / "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    return {
        "vq_checkpoint": root / "pretrained/VQVAE/net_last.pth",
        "mean": meta / "mean.npy",
        "std": meta / "std.npy",
        "evaluator_options": root / "checkpoints/t2m/Comp_v6_KLD005/opt.txt",
        "evaluator_checkpoint": root / "checkpoints/t2m/text_mot_match/model/finest.tar",
        "glove_vectors": root / "glove/our_vab_data.npy",
        "glove_indices": root / "glove/our_vab_idx.pkl",
        "glove_words": root / "glove/our_vab_words.pkl",
    }


def require_assets(asset_root, preparation_only=False):
    paths = asset_paths(asset_root)
    keys = ["vq_checkpoint", "mean", "std"] if preparation_only else paths.keys()
    missing = [str(paths[k]) for k in keys if not paths[k].is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    return paths


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vq(config, checkpoint, device="cuda"):
    import torch
    from models.vqvae import HumanVQVAE
    from utils.checkpoint import model_state

    cfg = SimpleNamespace(**config)
    net = HumanVQVAE(
        cfg,
        cfg.nb_code,
        cfg.code_dim,
        cfg.output_emb_width,
        cfg.down_t,
        cfg.stride_t,
        cfg.width,
        cfg.depth,
        cfg.dilation_growth_rate,
    )
    net.load_state_dict(
        model_state(torch.load(checkpoint, map_location="cpu", weights_only=True), "net"),
        strict=True,
    )
    return net.to(device).eval().requires_grad_(False)


def load_transformer(config, checkpoint, device="cuda"):
    import torch
    from models.t2m_trans import Text2Motion_Transformer
    from utils.checkpoint import model_state

    net = Text2Motion_Transformer(
        num_vq=config["nb_code"],
        embed_dim=config["embed_dim_gpt"],
        clip_dim=config["clip_dim"],
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        n_head=config["n_head_gpt"],
        drop_out_rate=config["drop_out_rate"],
        fc_rate=config["ff_rate"],
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(model_state(saved, "trans"), strict=True)
    return net.to(device).eval().requires_grad_(False), saved.get("iteration")
