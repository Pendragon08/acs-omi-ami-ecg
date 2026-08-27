"""Random seed utilities (aligned with 实施方案 section 29: seeds 2026-2030)."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEEDS = (2026, 2027, 2028, 2029, 2030)
