import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="WaveVerse-HMG transformer training",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## dataloader

    parser.add_argument(
        "--dataname", type=str, default="t2m", choices=["t2m"], help="dataset directory"
    )
    parser.add_argument("--batch-size", default=128, type=int, help="batch size")

    ## optimization
    parser.add_argument(
        "--total-iter", default=300000, type=int, help="number of total iterations to run"
    )
    parser.add_argument("--lr", default=0.0001, type=float, help="max learning rate")
    parser.add_argument(
        "--lr-scheduler",
        default=[150000],
        nargs="+",
        type=int,
        help="learning rate schedule (iterations)",
    )
    parser.add_argument("--gamma", default=0.05, type=float, help="learning rate decay")

    parser.add_argument("--weight-decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adam", "adamw"],
        help="disable weight decay on codebook",
    )

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument(
        "--mu", type=float, default=0.99, help="exponential moving average to update the codebook"
    )
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=165, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument(
        "--clip-dim", type=int, default=512, help="latent dimension in the clip feature"
    )
    parser.add_argument("--num-layers", type=int, default=4, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument(
        "--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding"
    )

    ## quantizer
    parser.add_argument(
        "--quantizer",
        type=str,
        default="ema_reset",
        choices=["ema", "orig", "ema_reset", "reset"],
        help="eps for optimal transport",
    )

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help="resume vq pth")
    parser.add_argument("--resume-trans", type=str, default=None, help="resume gpt pth")

    ## output directory
    parser.add_argument("--out-dir", type=str, default="output/", help="output directory")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="exp_debug",
        help="name of the experiment, will create a file inside out-dir",
    )
    parser.add_argument(
        "--vq-name",
        type=str,
        default="VQVAE",
        help="name of the generated dataset .npy, will create a file inside out-dir",
    )
    ## other
    parser.add_argument("--print-iter", default=200, type=int, help="print frequency")
    parser.add_argument("--eval-iter", default=10000, type=int, help="evaluation frequency")
    parser.add_argument("--seed", default=123, type=int, help="seed for initializing training. ")
    parser.add_argument("--if-maxtest", action="store_true", help="test in max")
    parser.add_argument("--pkeep", type=float, default=1.0, help="keep rate for gpt training")

    parser.add_argument(
        "--path-masking",
        action="store_true",
        help="Enable upstream contiguous path-coordinate masking",
    )
    parser.add_argument("--path-mask-rate", type=float, nargs=2, default=[0.5, 0.9])
    parser.add_argument("--path-mask-max-len", type=int, default=5)
    parser.add_argument("--path-mask-attempts", type=int, default=20)
    parser.add_argument("--path-mask-skip-prob", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-iter", type=int, default=1000)
    parser.add_argument(
        "--keep-iter",
        type=int,
        nargs="*",
        default=[],
        help="Keep weights at these iterations in addition to rotating checkpoints",
    )
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--pose-loss-weight",
        type=float,
        default=0.0,
        help="Weight of upstream decoded global-pose smooth-L1 loss",
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    print("parse known args")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unknown args: " + " ".join(unknown))

    if not (0 <= args.path_mask_rate[0] <= args.path_mask_rate[1] <= 1):
        parser.error("path-mask-rate must satisfy 0 <= min <= max <= 1")
    if (
        not 0 <= args.path_mask_skip_prob <= 1
        or min(args.path_mask_max_len, args.path_mask_attempts) < 1
    ):
        parser.error("Invalid path masking probability, length, or attempt count")
    if not 0 <= args.pose_loss_weight < float("inf"):
        parser.error("pose-loss-weight must be finite and nonnegative")
    return args
