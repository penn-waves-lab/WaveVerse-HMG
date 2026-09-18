# WaveVerse-HMG: Path-Conditioned Human Motion Generation

<a href="https://arxiv.org/abs/2508.12176"><img src="https://img.shields.io/badge/arXiv-2508.12176-b31b1b.svg" alt="arXiv"></a>
<a href="https://waves.seas.upenn.edu/projects/waveverse/"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page"></a>

WaveVerse-HMG is the **human motion generation** component accompanying
*Scalable RF Simulation in Generative 4D Worlds*. It generates human motion from
**text descriptions and spatial paths** using a path-conditioned autoregressive
transformer with path masking. This repository provides training and evaluation
on HumanML3D with a pretrained VQ motion tokenizer.
The RF simulator is available in [WaveVerse-Sim](https://github.com/penn-waves-lab/WaveVerse-Sim).

## 🛠️ Installation

Preparation, training, and evaluation require an NVIDIA GPU. The environment
recipe uses Python 3.8.20 and PyTorch 2.4.1 with CUDA 12.1. Run the following
commands from the repository root:

```bash
conda env create -f environment.yml
conda activate waveverse-hmg
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 📦 Dataset

Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
to prepare the dataset, then place it under `dataset/HumanML3D/` in the repository:

```text
dataset/HumanML3D/
├── new_joint_vecs/   # 263-dimensional motion features (.npy)
├── texts/            # Text annotations
├── train.txt
├── val.txt
└── test.txt
```

### 📥 Checkpoints and evaluation assets

Download the checkpoint archive from Google Drive and install it under `assets/`:

```bash
python tools/download_checkpoints.py
```

For an archive already downloaded locally:

```bash
python tools/download_checkpoints.py --archive WaveVerse-HMG-checkpoints.zip
```

Place the normalization statistics, evaluator options, and GloVe word vectors
under `assets/` separately; these are not included in the checkpoint-only archive.

```text
assets/
├── pretrained/
│   ├── VQVAE/net_last.pth
│   └── WaveVerse-HMG/net_best_fid.pth
├── checkpoints/t2m/
│   ├── VQVAEV3_CB1024_CMT_H1024_NRES3/meta/
│   │   ├── mean.npy
│   │   └── std.npy
│   ├── Comp_v6_KLD005/opt.txt
│   └── text_mot_match/model/finest.tar
└── glove/
    ├── our_vab_data.npy
    ├── our_vab_idx.pkl
    └── our_vab_words.pkl
```


## 🚀 Training

Training learns the motion transformer while keeping the pretrained VQ tokenizer frozen.

### 1. Prepare motion tokens

Generate the token sequences, reconstructed motion features, and prefix-position
caches used by the trainer. Use a new or empty cache directory.

```bash
python prepare_tokens.py
python tools/check_data.py
```

### 2. Train the motion transformer

```bash
python train.py
```

## 📊 Evaluation

Evaluate a trained transformer checkpoint:

```bash
python evaluate.py --checkpoint outputs/waveverse-hmg/net_best_fid.pth
```

To evaluate the downloaded original paper checkpoint:

```bash
python evaluate.py --checkpoint assets/pretrained/WaveVerse-HMG/net_best_fid.pth
```


## 🙏 Acknowledgements

This code builds on [T2M-GPT](https://github.com/Mael-zys/T2M-GPT). We thank the authors for sharing their code and resources.

## 📜 Citation

If you use WaveVerse-HMG in your research, please cite:

```bibtex
@inproceedings{zheng2026scalable,
  title={Scalable RF Simulation in Generative 4D Worlds},
  author={Zhiwei Zheng and Dongyin Hu and Mingmin Zhao},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```
