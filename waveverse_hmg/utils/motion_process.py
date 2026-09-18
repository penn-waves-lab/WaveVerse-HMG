import torch
import numpy as np
from utils.quaternion import qrot, qinv


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    """Add Y-axis rotation to local joints"""
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def resample_trajectory_64(points):
    """
    Resample a 2D trajectory of shape (N, 2) to exactly 64 points
    with a constant arc-length spacing.

    Args:
        points: (N, 2) NumPy array of xy-coordinates.

    Returns:
        sampled: (64, 2) NumPy array of the resampled points.
    """
    # 1) Segment vectors & lengths
    seg_vectors = points[1:] - points[:-1]  # shape (N-1, 2)
    seg_lengths = np.linalg.norm(seg_vectors, axis=1)  # shape (N-1,)

    # 2) Cumulative lengths (prefix sums), total arc length
    cum_lengths = np.cumsum(seg_lengths)  # shape (N-1,)
    total_length = cum_lengths[-1] if len(cum_lengths) > 0 else 0.0

    # Special case: If there's only 1 or 0 points, just pad or return as best we can
    if len(points) <= 1 or total_length == 0.0:
        return np.tile(points[0], (64, 1)) if len(points) > 0 else np.zeros((64, 2))

    # 3) Create 64 equally spaced arc-lengths from 0..total
    sample_distances = np.linspace(0, total_length, 64)  # shape (64,)

    # 4) For each sample distance, find which segment it belongs to
    #    np.searchsorted returns indices such that cum_lengths[idx-1] < distance <= cum_lengths[idx].
    #    We'll clamp to ensure idx never goes out of range
    seg_indices = np.searchsorted(cum_lengths, sample_distances, side="right")
    seg_indices = np.clip(seg_indices, 0, len(seg_vectors) - 1)  # shape (64,)

    # 5) Distance at the start of each chosen segment.  For segment i, that start is cum_lengths[i-1],
    #    except for i=0 (start=0).
    seg_starts = np.zeros_like(seg_indices, dtype=float)
    seg_starts[seg_indices > 0] = cum_lengths[seg_indices[seg_indices > 0] - 1]

    # 6) Fraction along each segment
    seg_dist = sample_distances - seg_starts
    seg_len_for_sample = seg_lengths[seg_indices] + 1e-15  # avoid div-by-zero
    frac = seg_dist / seg_len_for_sample  # shape (64,)

    # 7) Interpolate between the segment's endpoints:
    left_pts = points[seg_indices]  # shape (64, 2)
    right_pts = points[seg_indices + 1]  # shape (64, 2)

    sampled = left_pts + frac[:, None] * (right_pts - left_pts)  # shape (64, 2)

    return sampled  # in global


def chamfer_distance_masked(gt_paths, pred_paths, gt_lengths, pred_lengths):
    """
    Computes the Chamfer Distance for each variable-length ground truth and predicted path
    without using a for-loop.

    Args:
        gt_paths: NumPy array of shape (N, max_L_gt, 2), where max_L_gt is the longest GT path
        pred_paths: NumPy array of shape (N, max_L_pred, 2), where max_L_pred is the longest Pred path
        gt_lengths: NumPy array of shape (N,), indicating valid points per GT path
        pred_lengths: NumPy array of shape (N,), indicating valid points per predicted path

    Returns:
        avg_chamfer_distance: Scalar float, the average Chamfer Distance across all N pairs.
    """
    N, max_L_gt, _ = gt_paths.shape
    _, max_L_pred, _ = pred_paths.shape

    # Create masks for valid points
    gt_mask = np.arange(max_L_gt)[None, :] < gt_lengths[:, None]  # Shape: (N, max_L_gt)
    pred_mask = np.arange(max_L_pred)[None, :] < pred_lengths[:, None]  # Shape: (N, max_L_pred)

    # Compute pairwise distances (N, max_L_gt, max_L_pred)
    dist = np.linalg.norm(gt_paths[:, :, None, :] - pred_paths[:, None, :, :], axis=-1)

    # Apply masks: Set invalid distances to a large value so they do not contribute
    dist_masked = np.where(gt_mask[:, :, None] & pred_mask[:, None, :], dist, np.inf)

    # Compute Chamfer Distance
    chamfer_gt_to_pred = np.min(dist_masked, axis=2)  # (N, max_L_gt)
    chamfer_pred_to_gt = np.min(dist_masked, axis=1)  # (N, max_L_pred)

    # Mask out invalid distances
    chamfer_gt_to_pred[~gt_mask] = 0
    chamfer_pred_to_gt[~pred_mask] = 0

    # Compute sum of valid distances
    chamfer_gt_to_pred_sum = np.sum(chamfer_gt_to_pred, axis=1)  # (N,)
    chamfer_pred_to_gt_sum = np.sum(chamfer_pred_to_gt, axis=1)  # (N,)

    # Compute valid counts
    valid_gt_counts = np.sum(gt_mask, axis=1)  # (N,)
    valid_pred_counts = np.sum(pred_mask, axis=1)  # (N,)

    # Avoid division by zero
    valid_gt_counts[valid_gt_counts == 0] = 1
    valid_pred_counts[valid_pred_counts == 0] = 1

    # Compute mean Chamfer Distance per pair
    chamfer_gt_to_pred_mean = chamfer_gt_to_pred_sum / valid_gt_counts  # (N,)
    chamfer_pred_to_gt_mean = chamfer_pred_to_gt_sum / valid_pred_counts  # (N,)

    chamfer_distances = chamfer_gt_to_pred_mean + chamfer_pred_to_gt_mean  # (N,)

    # Compute the mean Chamfer Distance across all N pairs
    avg_chamfer_distance = np.mean(chamfer_distances)  # Scalar

    return avg_chamfer_distance
