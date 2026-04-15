import os

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    def __init__(self, root: str) -> None:
        super().__init__()

        assert os.path.isdir(root), f"{root} does not exist."
        self.root = root

        files = sorted([x for x in os.listdir(self.root) if x.endswith(".npz")])
        assert files, f"{root} is empty."
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, torch.Tensor):
            index = index.item()

        file_path = os.path.join(self.root, self.files[index]) 
        episode = np.load(file_path)

        latents = torch.as_tensor(episode["latents"], dtype=torch.float32)
        actions = torch.as_tensor(episode["actions"], dtype=torch.float32)

        return latents, actions