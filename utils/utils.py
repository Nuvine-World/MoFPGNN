import os
import random

import numpy as np
import torch


def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed=0):
    """Dedicated torch.Generator for DataLoader shuffling (see seed_everything)."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
