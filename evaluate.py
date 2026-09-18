"""Evaluate a trained path-conditioned model using recorded test seeds."""

import argparse
import json
import logging
import os
from pathlib import Path
import statistics

from tools.common import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_ASSET_ROOT,
    DEFAULT_EVAL_OUTPUT,
    configure,
    load_transformer,
    load_vq,
    read_config,
    require_assets,
    seed_all,
    sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    config = read_config(args.config)
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data, assets, _ = configure(args.data_root, args.asset_root)
    paths = require_assets(assets)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    os.environ["HMG_EVAL_SAMPLES"] = "1024"
    import torch
    import clip
    from dataset import dataset_TM_eval
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils import eval_trans, utils_model
    from utils.word_vectorizer import WordVectorizer

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA.")
    seed_all(123)
    net = load_vq(config, paths["vq_checkpoint"])
    trans, iteration = load_transformer(config, checkpoint)
    clip_model, _ = clip.load(
        "ViT-B/32", device="cuda", jit=False, download_root=os.environ.get("HMG_CLIP_CACHE")
    )
    clip.model.convert_weights(clip_model)
    clip_model.eval().requires_grad_(False)
    opt = get_opt(str(paths["evaluator_options"]), torch.device("cuda"))
    opt.checkpoints_dir = str(assets / "checkpoints")
    wrapper = EvaluatorModelWrapper(opt)
    vectorizer = WordVectorizer(str(assets / "glove"), "our_vab")
    provenance = {
        "checkpoint_sha256": sha256(checkpoint),
        "iteration": iteration,
        "asset_sha256": {key: sha256(p) for key, p in paths.items()},
        "split_sha256": sha256(data / (args.split + ".txt")),
        "torch": torch.__version__,
        "seeds": args.seeds,
        "num_workers": args.num_workers,
        "config": config,
        "split": args.split,
        "sample_cap": 1024,
        "batch_size": 32,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    results = []
    for seed in args.seeds:
        seed_all(seed)
        folder = output / ("seed_" + str(seed))
        folder.mkdir()
        loader = dataset_TM_eval.DATALoader(
            "t2m", args.split == "test", 32, vectorizer, num_workers=args.num_workers
        )
        (folder / "evaluation_subset.json").write_text(json.dumps(list(loader.dataset.name_list)))
        logger = logging.getLogger("Exp")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger = utils_model.get_logger(str(folder))
        eval_trans.evaluation_transformer(
            str(folder),
            loader,
            net,
            trans,
            logger,
            iteration or 0,
            1000,
            1000,
            0,
            100,
            0,
            0,
            0,
            100,
            clip_model,
            wrapper,
            save=False,
        )
        metrics = json.loads((folder / "metrics.jsonl").read_text().splitlines()[-1])
        assert metrics["samples"] == 1024
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        metrics["seed"] = seed
        results.append(metrics)
        (output / "results.json").write_text(json.dumps(results, indent=2))
    summary = {
        key: {
            "mean": statistics.mean(r[key] for r in results),
            "std": statistics.stdev(r[key] for r in results) if len(results) > 1 else 0.0,
        }
        for key in results[0]
        if key not in ["seed", "iteration", "samples"]
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "best_fid.json").write_text(
        json.dumps(min(results, key=lambda x: x["fid"]), indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
