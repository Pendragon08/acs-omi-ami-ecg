"""ACS multi-label Dataset.

Loads consolidated [N,1000,12] float32 arrays (built by
preprocessing/build_variants.py) via mmap and yields (x [1000,12], y [2]) with
y = [OMI, AMI].  Optional on-the-fly augmentation (train only).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.augment import augment


class ACSDataset(Dataset):
    def __init__(self, array_path, labels=None, augment=False, rng_seed=None,
                 return_mask=False, med_path=None, clin_path=None):
        self.array = np.load(array_path, mmap_mode="r")  # [N, 1000, 12]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augment_flag = augment
        self.return_mask = return_mask
        self.rng = np.random.default_rng(rng_seed)
        # Optional extra modalities (multimodal fusion):
        #   med  [N, 100, 12]  median beat (1s @100Hz)
        #   clin [N, clin_dim] clinical features (gender, age, ...)
        self.med = None if med_path is None else np.load(med_path, mmap_mode="r")
        self.clin = None if clin_path is None else np.load(clin_path, mmap_mode="r")

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx):
        # copy=True: mmap returns read-only views; we need a writable float32
        # array so torch.from_numpy produces a proper writable tensor.
        x = np.array(self.array[idx], dtype=np.float32, copy=True)  # [1000, 12]
        if self.augment_flag:
            x = augment(x, rng=self.rng)
        x = torch.from_numpy(np.ascontiguousarray(x))

        out = [x]
        if self.med is not None:
            med = np.array(self.med[idx], dtype=np.float32, copy=True)
            out.append(torch.from_numpy(np.ascontiguousarray(med)))
        if self.clin is not None:
            clin = np.array(self.clin[idx], dtype=np.float32, copy=True)
            out.append(torch.from_numpy(clin))

        y = torch.zeros(2, dtype=torch.float32)
        if self.labels is not None:
            y = torch.from_numpy(self.labels[idx].astype(np.float32))
        out.append(y)

        if self.return_mask:
            mask = torch.ones(x.shape[0], dtype=torch.float32)
            out.append(mask)
        # Order: [x, (med), (clin), y, (mask)]
        return tuple(out)
