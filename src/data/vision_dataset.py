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

        indices = []
        for chunk_id, chunk in enumerate(self.files):
            chunk = np.load(os.path.join(self.root, chunk), "r")
            n = chunk.shape[0]
            indices.extend((chunk_id, id) for id in range(n))
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            index = index.item()
        chunk_id, index = self.indices[index]

        file_path = os.path.join(self.root, self.files[chunk_id]) 
        chunk = np.load(file_path)

        obs = torch.as_tensor(np.array(chunk[index]), dtype=torch.float32).div_(255.0)

        return obs