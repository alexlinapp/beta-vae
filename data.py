import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as Transforms
import numpy as np
from pathlib import Path

from utils import download_dsprites


class dSprites(Dataset):
    def __init__(self, datasets_dir):
        datasets_dir = Path(datasets_dir)
        self.data_path = datasets_dir / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        self.dataset_zip = np.load(self.data_path, allow_pickle=True, encoding="bytes")
        self.imgs = self.dataset_zip["imgs"]
        self.latents_values = self.dataset_zip["latents_values"]
        self.latents_classes = self.dataset_zip["latents_classes"]
        self.metadata = self.dataset_zip["metadata"][()] # python indexing into 0D array

        self.length = self.imgs.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx]).to(torch.float32)
        return img


def get_dataloaders(cfg):
    download_dsprites(cfg.datasets_dir)
    dSprites_ds = dSprites(cfg.datasets_dir)
    train_ds, test_ds = random_split(dSprites_ds, [cfg.train_split, 1-cfg.train_split])

    train_loader = DataLoader(dataset=train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader
