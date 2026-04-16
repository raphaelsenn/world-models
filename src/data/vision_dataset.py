from typing import Callable
import os

import numpy as np

import torch
from torch.utils.data import Dataset


class VisionDataset(Dataset):
    """
    NOTE: The original paper collected 10'000 rollouts and trained for one epoch one those.
    However, since i don't have enough space, i relay viewer rollouts with heavy data augmentation.
    """ 
    def __init__(
            self, 
            root: str, 
            aug_transform: Callable | None = None
    ) -> None:
        super().__init__()

        assert os.path.isdir(root), f"{root} does not exist."
        self.root = root

        files = sorted([x for x in os.listdir(self.root) if x.endswith(".npy")])
        assert files, f"{root} is empty."
        self.files = files

        self.aug_transform = aug_transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            index = index.item()

        file_path = os.path.join(self.root, self.files[index])
        img = np.load(file_path)

        obs = torch.from_numpy(img).float().div_(255.0)

        if self.aug_transform is not None:
            obs = self.aug_transform(obs)

        return obs