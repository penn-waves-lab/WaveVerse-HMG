import os as _os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from utils.motion_process import recover_from_ric, resample_trajectory_64

_HERE = _os.path.dirname(_os.path.abspath(__file__))


from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


"""For use of training text-2-motion generative model"""


class Text2MotionDataset(data.Dataset):
    def __init__(
        self, dataset_name, is_test, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4
    ):

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        if dataset_name == "t2m":
            self.data_root = _os.environ.get(
                "HMG_DATA_ROOT",
                _os.path.normpath(_os.path.join(_HERE, "..", "dataset", "HumanML3D")),
            )
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = _os.path.normpath(
                _os.path.join(
                    _HERE, "..", "checkpoints", "t2m", "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta"
                )
            )
        elif dataset_name == "kit":
            self.data_root = _os.environ.get(
                "HMG_DATA_ROOT", _os.path.normpath(_os.path.join(_HERE, "..", "dataset", "KIT-ML"))
            )
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 21
            self.max_motion_length = 196
            self.meta_dir = _os.path.normpath(
                _os.path.join(
                    _HERE, "..", "checkpoints", "kit", "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta"
                )
            )

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

        if is_test:
            split_file = pjoin(self.data_root, "test.txt")
        else:
            split_file = pjoin(self.data_root, "val.txt")

        min_motion_len = 40 if self.dataset_name == "t2m" else 24

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
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            raise ValueError("should not be here")
                            # break

                if flag:
                    data_dict[name] = {"motion": motion, "length": len(motion), "text": text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception:
                pass

            if len(new_name_list) >= int(_os.environ.get("HMG_EVAL_SAMPLES", "1024")):  # 1024
                break

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.mean_torch = torch.tensor(mean)
        self.std_torch = torch.tensor(std)
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_torch(self, data):
        return data * self.std_torch.to(data.device) + self.mean_torch.to(data.device)

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        joints = recover_from_ric(torch.tensor(motion), 22).numpy()
        root = joints[:, 0, [0, 2]]
        path = resample_trajectory_64(root)
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0
            )

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            name,
            path,
        )


def DATALoader(dataset_name, is_test, batch_size, w_vectorizer, num_workers=8, unit_length=4):

    val_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return val_loader
