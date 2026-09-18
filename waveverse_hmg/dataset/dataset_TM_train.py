import os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from utils.motion_process import recover_from_ric, resample_trajectory_64

"""For use of training text-2-motion generative model"""


class Text2MotionDataset(data.Dataset):
    def __init__(
        self, dataset_name, feat_bias=5, unit_length=4, codebook_size=1024, tokenizer_name=None
    ):

        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == "t2m":
            self.data_root = os.environ.get("HMG_DATA_ROOT", "./dataset/HumanML3D")
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 26 if unit_length == 8 else 51
        elif dataset_name == "kit":
            self.data_root = os.environ.get("HMG_DATA_ROOT", "./dataset/KIT-ML")
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 21
            self.max_motion_length = 26 if unit_length == 8 else 51

        split_file = pjoin(self.data_root, "train.txt")

        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s.npy" % name,
                    )
                )
                feat = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s_recfeat.npy" % name,
                    )
                )
                feat_pos = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s_recincrepos.npy" % name,
                    )
                )
                feat_dropfirst = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s_recfeat_dropfirstcode.npy" % name,
                    )
                )
                feat_dropfirst_pos = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s_recincrepos_dropfirstcode.npy" % name,
                    )
                )
                feat_droplast = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s_recfeat_droplastcode.npy" % name,
                    )
                )
                feat_droplast_pos = np.load(
                    pjoin(
                        os.environ.get("HMG_TOKEN_ROOT", pjoin(self.data_root, tokenizer_name)),
                        "%s_recincrepos_droplastcode.npy" % name,
                    )
                )

                # Read text
                with cs.open(pjoin(self.text_dir, name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split("#")
                            caption = line_split[0]
                            t_tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                raise ValueError("should not be here")
                        except:
                            pass

                if flag:
                    data_dict[name] = {
                        "m_token_list": m_token_list,
                        "text": text_data,
                        "feat": feat,
                        "feat_pos": feat_pos,
                        "feat_dropfirst": feat_dropfirst,
                        "feat_dropfirst_pos": feat_dropfirst_pos,
                        "feat_droplast": feat_droplast,
                        "feat_droplast_pos": feat_droplast_pos,
                    }
                    new_name_list.append(name)
            except:
                pass
            limit = int(os.environ.get("HMG_TRAIN_SAMPLES", "0"))
            if limit and len(new_name_list) >= limit:
                break
        if not data_dict:
            raise RuntimeError("No training samples found; check HMG_DATA_ROOT and HMG_TOKEN_ROOT.")
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list, raw_feat, feat_pos = (
            data["m_token_list"],
            data["text"],
            data["feat"],
            data["feat_pos"],
        )
        feat_dropfirst, dropfirst_pos, feat_droplast, droplast_pos = (
            data["feat_dropfirst"],
            data["feat_dropfirst_pos"],
            data["feat_droplast"],
            data["feat_droplast_pos"],
        )
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption = text_data["caption"]

        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
                feat = feat_droplast
                pos = droplast_pos
            else:
                m_tokens = m_tokens[1:]
                feat = feat_dropfirst
                pos = dropfirst_pos
        else:
            feat = raw_feat
            pos = feat_pos

        m_tokens_len = m_tokens.shape[0]
        pos = pos[0]
        assert pos.shape[0] == m_tokens_len

        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate(
                [
                    m_tokens,
                    np.ones((1), dtype=int) * self.mot_end_idx,
                    np.ones((self.max_motion_length - 1 - m_tokens_len), dtype=int)
                    * self.mot_pad_idx,
                ],
                axis=0,
            )
            m_pos = np.concatenate(
                [
                    pos,
                    np.ones((1, 2), dtype=int) * self.mot_end_idx,
                    np.ones((self.max_motion_length - 1 - m_tokens_len, 2), dtype=int)
                    * self.mot_pad_idx,
                ]
            )
        else:
            m_tokens = np.concatenate(
                [m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0
            )
            m_pos = np.concatenate([pos, np.ones((1, 2), dtype=int) * self.mot_end_idx], axis=0)

        joints = recover_from_ric(torch.tensor(feat, dtype=torch.float), 22).numpy()
        root = joints[0][:, 0, [0, 2]]
        path = resample_trajectory_64(root)
        return caption, m_tokens.reshape(-1), m_tokens_len, path, m_pos


def DATALoader(
    dataset_name, batch_size, codebook_size, tokenizer_name, unit_length=4, num_workers=8
):

    train_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(
            dataset_name,
            codebook_size=codebook_size,
            tokenizer_name=tokenizer_name,
            unit_length=unit_length,
        ),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
