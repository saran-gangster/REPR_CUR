import os
import requests
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

# --- Tokenizers --------------------------------------------------------------

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

class ByteTokenizer:
    def __init__(self):
        # 256 byte values
        self.vocab_size = 256

    def encode(self, s: str):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        try:
            return bytes(ids).decode("utf-8", errors="ignore")
        except Exception:
            return ""

class TiktokenTokenizer:
    def __init__(self, name: str = "cl100k_base"):
        try:
            import tiktoken  # type: ignore
        except Exception as e:
            raise ImportError("tiktoken not installed") from e
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, s: str):
        return self.enc.encode(s)

    def decode(self, ids):
        return self.enc.decode(ids)

# --- Dataset ----------------------------------------------------------------

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

# --- DataModule -------------------------------------------------------------

class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/", block_size: int = 512, batch_size: int = 32,
                 num_workers: int = 2, download_url: str = None,
                 tokenizer_type: str = "char",
                 tokenizer_name: str = "cl100k_base",
                 use_synthetic: bool = False,
                 synthetic_vocab_size: int = 16,
                 synthetic_segment_len: int = 192,
                 synthetic_num_segments: int = 3):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download_url = download_url

        # new controls
        self.tokenizer_type = tokenizer_type
        self.tokenizer_name = tokenizer_name
        self.use_synthetic = use_synthetic
        self.synthetic_vocab_size = int(synthetic_vocab_size)
        self.synthetic_segment_len = int(synthetic_segment_len)
        self.synthetic_num_segments = int(synthetic_num_segments)

        self.tokenizer = None
        self.vocab_size = None

    def prepare_data(self):
        if self.use_synthetic:
            return  # nothing to download
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "tiny_shakespeare.txt")
        if not os.path.exists(file_path):
            assert self.download_url is not None, "Provide a download_url for Tiny Shakespeare."
            r = requests.get(self.download_url)
            r.raise_for_status()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(r.text)

    def _load_text(self) -> str:
        file_path = os.path.join(self.data_dir, "tiny_shakespeare.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_tokenizer(self, text: str):
        ttype = (self.tokenizer_type or "char").lower()
        if ttype == "char":
            tok = CharTokenizer(text)
        elif ttype == "byte":
            tok = ByteTokenizer()
        elif ttype == "tiktoken":
            try:
                tok = TiktokenTokenizer(self.tokenizer_name)
            except Exception:
                # graceful fallback to char if tiktoken missing
                tok = CharTokenizer(text)
        else:
            tok = CharTokenizer(text)
        return tok
    def _make_synthetic_data(self, total_tokens: int) -> np.ndarray:
        # Build a long 1D array with repeated segments, each segment uses a unique token id.
        # Reserve 0 as a separator token (optional).
        V = max(self.synthetic_vocab_size, self.synthetic_num_segments + 1)
        seg_len = max(8, self.synthetic_segment_len)
        n_segs = self.synthetic_num_segments
        # Create a repeating pattern [SEP, 1,1,..., SEP, 2,2,..., SEP, 3,3,...]
        seq = []
        token_ids = list(range(1, n_segs + 1))
        while len(seq) < total_tokens:
            for tid in token_ids:
                seq.append(0)  # SEP
                seq.extend([tid] * seg_len)
                if len(seq) >= total_tokens:
                    break
        arr = np.array(seq[:total_tokens], dtype=np.int32)
        # Ensure vocab_size is set
        self.vocab_size = V
        return arr

    def setup(self, stage: Optional[str] = None):
        if self.use_synthetic:
            # 1M tokens gives plenty of chunks; adjust if memory-constrained
            total_tokens = max(10 * self.block_size, 1_000_000)
            data = self._make_synthetic_data(total_tokens)
            # 90/10 split
            split_idx = int(0.9 * len(data))
            self.train_data = data[:split_idx]
            self.val_data = data[split_idx:]
            self.train_ds = CharDataset(self.train_data, self.block_size)
            self.val_ds = CharDataset(self.val_data, self.block_size)
            # tokenizer is not used for synthetic; but expose vocab_size
            return

        text = self._load_text()
        self.tokenizer = self._build_tokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size

        data = np.array(self.tokenizer.encode(text), dtype=np.int32)
        # 90/10 split
        split_idx = int(0.9 * len(data))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

        self.train_ds = CharDataset(self.train_data, self.block_size)
        self.val_ds = CharDataset(self.val_data, self.block_size)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
        if self.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
        if self.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(self.val_ds, **kwargs)