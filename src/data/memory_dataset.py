import os

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    def __init__(self, root: str, seq_len: int=64) -> None:
        super().__init__()

        assert os.path.isdir(root), f"{root} does not exist."
        self.root = root
        self.seq_len = seq_len

        files = sorted([x for x in os.listdir(self.root) if x.endswith(".npz")])
        assert files, f"{root} is empty."

        self.files = []
        self.lengths = []

        for file in files:
            file_path = os.path.join(self.root, file)

            with np.load(file_path) as episode:
                ep_len = len(episode["latents"])

            if ep_len >= self.seq_len + 1:
                self.files.append(file)
                self.lengths.append(ep_len)
        
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, torch.Tensor):
            index = index.item()

        file_path = os.path.join(self.root, self.files[index]) 
        episode = np.load(file_path)
        ep_len = self.lengths[index]

        max_start = ep_len - self.seq_len - 1
        start = np.random.randint(0, max_start)
        end = start + self.seq_len + 1

        latents = torch.as_tensor(
            episode["latents"][start:end], 
            dtype=torch.float32
        )
        actions = torch.as_tensor(
            episode["actions"][start:end], 
            dtype=torch.float32
        )

        return latents, actions