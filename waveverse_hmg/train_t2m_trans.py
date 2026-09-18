import os
import random
import torch
import numpy as np

from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
from utils.motion_process import recover_from_ric

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from utils.checkpoint import model_state
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from utils.path_masking import mask_paths_no_for_loop
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
import wandb

warnings.filterwarnings("ignore")


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
args.source_branch_commit = "4b6caa85183c993fd971d50c135e6640d64e0edb"
args.optimizer_betas = [0.5, 0.9]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
args.vq_dir = os.path.join(
    "./dataset/KIT-ML" if args.dataname == "kit" else "./dataset/HumanML3D", f"{args.vq_name}"
)
os.makedirs(args.out_dir, exist_ok=True)


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
wandb.init(
    project="waveverse-hmg",
    dir=args.out_dir,
    config=vars(args),
    name=args.exp_name,
    mode=os.environ.get("WANDB_MODE", "offline"),
)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
with open(pjoin(args.out_dir, "config.json"), "w") as config_file:
    json.dump(vars(args), config_file, indent=2)

##### ---- Dataloader ---- #####


from utils.word_vectorizer import WordVectorizer

asset_root = os.environ.get("HMG_ASSET_ROOT", ".")
w_vectorizer = WordVectorizer(pjoin(asset_root, "glove"), "our_vab")
val_loader = dataset_TM_eval.DATALoader(
    args.dataname, args.eval_split == "test", 32, w_vectorizer, num_workers=args.num_workers
)
with open(pjoin(args.out_dir, "evaluation_subset.json"), "w") as subset_file:
    json.dump(
        {"split": args.eval_split, "names": list(val_loader.dataset.name_list)},
        subset_file,
        indent=2,
    )

dataset_opt_path = (
    "checkpoints/kit/Comp_v6_KLD005/opt.txt"
    if args.dataname == "kit"
    else "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
)

wrapper_opt = get_opt(pjoin(asset_root, dataset_opt_path), torch.device("cuda"))
wrapper_opt.checkpoints_dir = pjoin(asset_root, "checkpoints")
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load(
    "ViT-B/32",
    device=torch.device("cuda"),
    jit=False,
    download_root=os.environ.get("HMG_CLIP_CACHE"),
)  # Must set jit=False for training
clip.model.convert_weights(
    clip_model
)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(
    args,  ## use args to define different parameters in different quantizers
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
)


trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=args.embed_dim_gpt,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,
)


print("loading checkpoint from {}".format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location="cpu")
net.load_state_dict(model_state(ckpt, "net"), strict=True)
for param in net.parameters():
    param.requires_grad = False
net.eval()
net.cuda()

if args.resume_trans is not None:
    print("loading transformer checkpoint from {}".format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location="cpu")
    trans_encoder.load_state_dict(model_state(ckpt, "trans"), strict=True)
trans_encoder.train()
trans_encoder.cuda()

print("set up optimizer and scheduler")
##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args.lr_scheduler, gamma=args.gamma
)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0.0, 0.0
avg_path_mask_fraction = 0.0
avg_pose_loss = 0.0
right_num = 0
nb_sample_train = 0


print("creating the dataloader")
# set batch_size to 1 if getting code for second-stage training
train_loader = dataset_TM_train.DATALoader(
    args.dataname,
    args.batch_size,
    args.nb_code,
    args.vq_name,
    unit_length=2**args.down_t,
    num_workers=args.num_workers,
)
if len(train_loader) == 0:
    raise RuntimeError("Training loader is empty: check HumanML3D and reconstructed token caches.")
logger.info(
    "Dataset sizes: train=%d eval_split=%s eval=%d",
    len(train_loader.dataset),
    args.eval_split,
    len(val_loader.dataset),
)
train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Training ---- #####
best_end_error, best_fid, best_iter = 1000.0, 1000.0, 0
best_div, best_top1, best_top2, best_top3, best_matching = 100.0, 0.0, 0.0, 0.0, 100.0
if not args.skip_initial_eval:
    (
        best_end_error,
        best_fid,
        best_iter,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        logger,
    ) = eval_trans.evaluation_transformer(
        args.out_dir,
        val_loader,
        net,
        trans_encoder,
        logger,
        0,
        best_end_error,
        best_fid,
        best_iter,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        clip_model=clip_model,
        eval_wrapper=eval_wrapper,
    )

print("start the training")
while nb_iter < args.total_iter:

    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, paths, m_pos = batch

    m_tokens, m_tokens_len, paths, m_pos = (
        m_tokens.cuda(),
        m_tokens_len.cuda(),
        paths.cuda().float(),
        m_pos.cuda().float(),
    )
    bs = m_tokens.shape[0]
    target = m_tokens  # (bs, 26)
    target = target.cuda()

    text = clip.tokenize(clip_text, truncate=True).cuda()

    with torch.no_grad():
        feat_clip_text = clip_model.encode_text(text).float()

    input_index = target[:, :-1]
    m_pos = m_pos[:, :-1, :]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape, device=input_index.device))
    else:
        mask = torch.bernoulli(
            args.pkeep * torch.ones(input_index.shape, device=input_index.device)
        )
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask * input_index + (1 - mask) * r_indices

    if args.path_masking:
        paths, path_mask = mask_paths_no_for_loop(
            paths,
            mask_rate_range=args.path_mask_rate,
            max_mask_len=args.path_mask_max_len,
            max_attempts=args.path_mask_attempts,
            no_random_prob=args.path_mask_skip_prob,
        )
        avg_path_mask_fraction += (1.0 - path_mask.mean()).item()

    cls_pred = trans_encoder(a_indices, feat_clip_text, paths, m_pos)
    cls_pred = cls_pred.contiguous()

    loss_cls = 0.0
    batch_pose_loss = 0.0
    for i in range(bs):
        # loss function     (26), (26, 513)
        cls_pred_token = cls_pred[i][64 : 2 * m_tokens_len[i] + 1 + 64 : 2]
        cls_tar_token = target[i][: m_tokens_len[i] + 1]
        loss_cls += loss_ce(cls_pred_token, cls_tar_token) / bs

        # Accuracy
        probs = torch.softmax(cls_pred_token, dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)

        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (
            (cls_pred_index.flatten(0) == target[i][: m_tokens_len[i] + 1].flatten(0)).sum().item()
        )

        # Active in upstream corl 4972b0f and path 4b6caa8. Keep gradients
        # through the frozen decoder into the straight-through token samples.
        if args.pose_loss_weight > 0:
            pred_idx_onehot = torch.nn.functional.gumbel_softmax(
                cls_pred_token[:-1, :-1], tau=1, hard=True
            )
            gt_idx = cls_tar_token[:-1]
            pred_pose = net.forward_onehot_decoder(pred_idx_onehot)
            gt_pose = net.forward_decoder(gt_idx)
            pred_denorm = val_loader.dataset.inv_transform_torch(pred_pose)
            gt_denorm = val_loader.dataset.inv_transform_torch(gt_pose)
            pred_xyz = recover_from_ric(pred_denorm.float(), 22)
            gt_xyz = recover_from_ric(gt_denorm.float(), 22)
            pose_error = (
                torch.nn.functional.smooth_l1_loss(pred_xyz, gt_xyz) / bs * args.pose_loss_weight
            )
            loss_cls += pose_error
            batch_pose_loss += pose_error.detach().item()

    ## global loss
    if not torch.isfinite(loss_cls):
        raise RuntimeError("Non-finite training loss at iteration %d" % nb_iter)
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    avg_pose_loss += batch_pose_loss
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter == 0:
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        wandb.log(
            {
                "Loss/train": avg_loss_cls,
                "Loss/pose_weighted": avg_pose_loss / args.print_iter,
                "ACC/train": avg_acc,
                "Path/masked_fraction": avg_path_mask_fraction / args.print_iter,
            },
            step=nb_iter,
        )
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}, Path masked. {avg_path_mask_fraction / args.print_iter:.4f}, Pose loss. {avg_pose_loss / args.print_iter:.5f}"
        logger.info(msg)
        avg_path_mask_fraction = 0.0
        avg_loss_cls = 0.0
        avg_pose_loss = 0.0
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.save_iter == 0 or nb_iter == args.total_iter:
        state = {
            "trans": trans_encoder.state_dict(),
            "iteration": nb_iter,
            "config": vars(args),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        checkpoint_tmp = pjoin(args.out_dir, "training_state.tmp")
        torch.save(state, checkpoint_tmp)
        os.replace(checkpoint_tmp, pjoin(args.out_dir, "training_state.pth"))
        torch.save(
            {"trans": trans_encoder.state_dict(), "iteration": nb_iter, "config": vars(args)},
            pjoin(args.out_dir, "net_last.pth"),
        )

    if nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
        (
            best_end_error,
            best_fid,
            best_iter,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            logger,
        ) = eval_trans.evaluation_transformer(
            args.out_dir,
            val_loader,
            net,
            trans_encoder,
            logger,
            nb_iter,
            best_end_error,
            best_fid,
            best_iter,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            clip_model=clip_model,
            eval_wrapper=eval_wrapper,
        )

    if nb_iter in args.keep_iter:
        torch.save(
            {"trans": trans_encoder.state_dict(), "iteration": nb_iter, "config": vars(args)},
            pjoin(args.out_dir, "net_iter_%d.pth" % nb_iter),
        )

    if nb_iter == args.total_iter:
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}; Best_EndingError. {best_end_error}"
        logger.info(msg_final)
        break
wandb.finish()
