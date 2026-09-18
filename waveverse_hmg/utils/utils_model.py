import torch.optim as optim
import logging
import os
import sys


def get_logger(out_dir):
    logger = logging.getLogger("Exp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


## Optimizer
def initial_optim(lr, weight_decay, net, optimizer):
    optimizer_class = {"adam": optim.Adam, "adamw": optim.AdamW}[optimizer]
    return optimizer_class(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
