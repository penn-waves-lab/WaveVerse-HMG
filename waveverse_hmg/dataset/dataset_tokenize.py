import os as _os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, window_size=64, unit_length=8):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name == "t2m" else 24

        if dataset_name == "t2m":
            self.data_root = _os.environ.get("HMG_DATA_ROOT", "./dataset/HumanML3D")
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
        elif dataset_name == "kit":
            self.data_root = _os.environ.get("HMG_DATA_ROOT", "./dataset/KIT-ML")
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 21
            self.max_motion_length = 196
            self.meta_dir = "checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"

        if _os.environ.get("HMG_ASSET_ROOT"):
            self.meta_dir = pjoin(
                _os.environ["HMG_ASSET_ROOT"],
                "checkpoints",
                dataset_name,
                "VQVAEV3_CB1024_CMT_H1024_NRES3",
                "meta",
            )

        mean = np.load(pjoin(self.meta_dir, "mean.npy"))
        std = np.load(pjoin(self.meta_dir, "std.npy"))

        split_file = pjoin(self.data_root, "train.txt")

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                data_dict[name] = {"motion": motion, "length": len(motion), "name": name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data["motion"], data["length"]

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name


def DATALoader(dataset_name, batch_size=1, num_workers=8, unit_length=4):

    train_dataset = VQMotionDataset(dataset_name, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    return train_loader, train_dataset
