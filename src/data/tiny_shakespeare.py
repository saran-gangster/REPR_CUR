import os
import json
import math
import requests
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

class CharDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data  # int32 array of token ids
        self.block_size = block_size

    def __len__(self):
        # number of blocks we can sample (roughly)
        return max(1, len(self.data) // self.block_size)

    def __getitem__(self, idx):
        # sample a random chunk
        start = np.random.randint(0, len(self.data) - self.block_size - 1)
        x = self.data[start : start + self.block_size]
        # JEPA doesn't need targets; we just feed x to get hidden states
        return torch.tensor(x, dtype=torch.long)

class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/", block_size: int = 512, batch_size: int = 32,
                 num_workers: int = 2, download_url: str = None):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download_url = download_url
        self.tokenizer = None
        self.vocab_size = None

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "tiny_shakespeare.txt")
        if not os.path.exists(file_path):
            assert self.download_url is not None, "Provide a download_url for Tiny Shakespeare."
            r = requests.get(self.download_url)
            r.raise_for_status()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(r.text)

    def setup(self, stage: Optional[str] = None):
        file_path = os.path.join(self.data_dir, "tiny_shakespeare.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = CharTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size

        data = np.array(self.tokenizer.encode(text), dtype=np.int32)
        # 90/10 split
        split_idx = int(0.9 * len(data))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

        self.train_ds = CharDataset(self.train_data, self.block_size)
        self.val_ds = CharDataset(self.val_data, self.block_size)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True)