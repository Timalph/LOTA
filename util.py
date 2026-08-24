import torch
import numpy as np
import random
import os
import json


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        cur_lr = lr
    return cur_lr


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

from torch import nn

def bceLoss():
    return nn.BCEWithLogitsLoss()


def crossEntropyLoss():
    return nn.CrossEntropyLoss()
def mseLoss():
    return nn.MSELoss()


def save_config(config, directory):
    """Persist the run configuration as JSON alongside model checkpoints,
    so a checkpoint always carries the settings it was trained with."""
    config_path = os.path.join(directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    return config_path


def load_training_config(checkpoint_path):
    """Load the config.json saved alongside a checkpoint (same directory
    as checkpoint_path). Returns None if no config.json is found."""
    config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r') as f:
        return json.load(f)