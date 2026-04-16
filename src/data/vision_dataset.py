import os

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class VisionDataset(Dataset):
    def __init__(self, root: str) -> None:
        super().__init__()

        assert os.path.isdir(root), f"{root} does not exist."
        self.root = root

        files = sorted([x for x in os.listdir(self.root) if x.endswith(".npy")])
        assert files, f"{root} is empty."
        self.files = files

        self.indices = []
        self.chunks = []

        for chunk_id, chunk_file in enumerate(self.files):
            chunk = np.load(os.path.join(self.root, chunk_file), mmap_mode="r")
            self.chunks.append(chunk)
            n = chunk.shape[0]
            self.indices.extend((chunk_id, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            index = index.item()

        chunk_id, local_index = self.indices[index]
        obs = torch.from_numpy(self.chunks[chunk_id][local_index].copy()).float().div_(255.0)
        return obs