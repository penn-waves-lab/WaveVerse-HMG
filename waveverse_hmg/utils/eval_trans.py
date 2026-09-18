import json
import os

import clip
import numpy as np
import torch
from scipy import linalg
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from utils.motion_process import recover_from_ric, resample_trajectory_64, chamfer_distance_masked


def calculate_skating_ratio(motions):
    thresh_height = 0.05  # 10
    fps = 20.0
    thresh_vel = 0.50  # 20 cm /s
    avg_window = 5  # frames

    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = (
        np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1], axis=2) * fps
    )  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(
        verts_feet_plane_vel, axis=-1, size=avg_window, mode="constant", origin=0
    )

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and(
        (verts_feet_height[:, :, :-1] < thresh_height),
        (verts_feet_height[:, :, 1:] < thresh_height),
    )  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel


@torch.no_grad()
def evaluation_transformer(
    out_dir,
    val_loader,
    net,
    trans,
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
    clip_model,
    eval_wrapper,
    save=True,
):
    """Evaluate one pass, retaining the original metric and sampling order."""
    trans.eval()
    nb_sample = 0
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    all_gt_paths = []
    all_gt_roots = []
    all_gt_length = []
    all_gt_end = []
    all_pred_paths = []
    all_pred_roots = []
    all_pred_length = []
    all_pred_end = []
    skate_ratio_sum = 0
    bone_length_var_all = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, paths = (
            batch
        )
        pred_paths = np.zeros_like(paths)
        all_gt_paths.append(paths)
        paths = paths.cuda().float()
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model.encode_text(text).float()
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()
        for k in range(bs):
            index_motion = trans.sample(
                feat_clip_text[k : k + 1], paths[k : k + 1], net, val_loader.dataset, False
            )
            pred_pose = net.forward_decoder(index_motion)
            cur_len = pred_pose.shape[1]
            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k : k + 1, :cur_len] = pred_pose[:, :seq]
            pred_denorm = val_loader.dataset.inv_transform(
                pred_pose_eval[k : k + 1].detach().cpu().numpy()
            )
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            cur_pred_root = pred_xyz[0][:196, 0, [0, 2]].cpu().numpy()
            cur_pred_path = resample_trajectory_64(cur_pred_root[: pred_len[k]])
            pred_paths[k] = cur_pred_path
            all_pred_roots.append(cur_pred_root[None, ...])
            all_pred_end.append(cur_pred_root[pred_len[k] - 1 : pred_len[k]])
            valid_pred_xyz = pred_xyz[:, : pred_len[k]]
            skate_ratio, skate_vel = calculate_skating_ratio(valid_pred_xyz.permute(0, 2, 3, 1))
            skate_ratio_sum += skate_ratio.sum()
            bone_indices = [[1, 4, 7], [2, 5, 8], [16, 18, 20], [17, 19, 21]]
            bone_length_var = 0
            for indices in bone_indices:
                bone_lengths = torch.norm(
                    valid_pred_xyz[:, :, indices[1], :] - valid_pred_xyz[:, :, indices[0], :], dim=2
                )
                bone_lengths2 = torch.norm(
                    valid_pred_xyz[:, :, indices[2], :] - valid_pred_xyz[:, :, indices[1], :], dim=2
                )
                bone_length_var += torch.var(bone_lengths * 100) + torch.var(bone_lengths2 * 100)
            bone_length_var = bone_length_var / len(bone_indices) / 2
            bone_length_var_all += bone_length_var.item()
        all_pred_length.append(pred_len)
        all_pred_paths.append(pred_paths)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len
        )
        pose = pose.cuda().float()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)
        cur_gt_root = pose_xyz[:, :, 0, [0, 2]].cpu().numpy()
        cur_gt_end = cur_gt_root[np.arange(pose.shape[0]), m_length - 1]
        all_gt_roots.append(cur_gt_root)
        all_gt_length.append(m_length)
        all_gt_end.append(cur_gt_end)
        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision += temp_R
        matching_score_pred += temp_match
        nb_sample += bs
    skate_ratio_avg = skate_ratio_sum / nb_sample
    print("skate ratio: ", skate_ratio_avg)
    bone_length_var_avg = bone_length_var_all / nb_sample
    print("bone length variance: ", bone_length_var_avg)
    all_gt_paths = torch.cat(all_gt_paths).numpy()
    all_pred_paths = np.concatenate(all_pred_paths)
    all_gt_length = torch.cat(all_gt_length).numpy()
    all_pred_length = torch.cat(all_pred_length).numpy()
    all_gt_roots = np.concatenate(all_gt_roots)
    all_pred_roots = np.concatenate(all_pred_roots)
    all_pred_end = np.concatenate(all_pred_end)
    all_gt_end = np.concatenate(all_gt_end)
    chamfer_path_error = np.mean(np.linalg.norm(all_gt_paths - all_pred_paths, axis=2))
    chamfer_root_error = chamfer_distance_masked(
        all_gt_roots, all_pred_roots, all_gt_length, all_pred_length
    )
    chamfer_root_path = chamfer_distance_masked(
        all_gt_paths, all_pred_roots, np.ones(all_gt_paths.shape[0]) * 64, all_pred_length
    )
    end_error = np.mean(np.linalg.norm(all_gt_end - all_pred_end, axis=1))
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    diversity_real = calculate_diversity(motion_annotation_np, min(300, nb_sample - 1))
    diversity = calculate_diversity(motion_pred_np, min(300, nb_sample - 1))
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    metrics = {
        "iteration": nb_iter,
        "samples": nb_sample,
        "fid": float(fid),
        "diversity": float(diversity),
        "top1": float(R_precision[0]),
        "top2": float(R_precision[1]),
        "top3": float(R_precision[2]),
        "matching_score": float(matching_score_pred),
        "path_error": float(chamfer_path_error),
        "trajectory_to_path": float(chamfer_root_path),
        "trajectory_error": float(chamfer_root_error),
        "ending_error": float(end_error),
        "skating_ratio": float(skate_ratio_avg),
        "bone_length_variance": float(bone_length_var_avg),
    }
    with open(os.path.join(out_dir, "metrics.jsonl"), "a") as metrics_file:
        metrics_file.write(json.dumps(metrics) + "\n")
    logger.info(
        "%s metrics: %s",
        "Test" if val_loader.dataset.is_test else "Validation",
        json.dumps(metrics),
    )
    print("result")
    print(
        {
            "Test/FID": fid,
            "Test/Diversity": diversity,
            "Test/top1": R_precision[0],
            "Test/top2": R_precision[1],
            "Test/top3": R_precision[2],
            "Test/matching_score": matching_score_pred,
            "Test/chamfer_path2path": chamfer_path_error,
            "Test/chamfer_traj2path": chamfer_root_path,
            "Test/chamfer_traj2traj": chamfer_root_error,
            "Test/ending_error": end_error,
        }
    )
    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = (fid, nb_iter)
        if save:
            torch.save(
                {"trans": trans.state_dict(), "iteration": nb_iter},
                os.path.join(out_dir, "net_best_fid.pth"),
            )
    if end_error < best_end_error:
        msg = f"--> --> \t EndError Improved from {best_end_error:.5f} to {end_error:.5f} !!!"
        logger.info(msg)
        best_end_error = end_error
        if save:
            torch.save(
                {"trans": trans.state_dict(), "iteration": nb_iter},
                os.path.join(out_dir, "net_best_endError.pth"),
            )
    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    if save:
        torch.save(
            {"trans": trans.state_dict(), "iteration": nb_iter},
            os.path.join(out_dir, "net_last.pth"),
        )
    trans.train()
    return (
        best_end_error,
        best_fid,
        best_iter,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        logger,
    )


def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; " "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov
