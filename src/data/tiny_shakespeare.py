import os
import requests
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

# --- Tokenizer --------------------------------------------------------------

class HFTokenizer:
    def __init__(self, tokenizer_json_path: str):
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as e:
            raise ImportError("Please `pip install tokenizers` to use the HF BPE tokenizer.") from e
        self.tk = Tokenizer.from_file(tokenizer_json_path)
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, s: str):
        return self.tk.encode(s).ids

    def decode(self, ids):
        try:
            return self.tk.decode(ids)
        except Exception:
            return ""

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
        return torch.tensor(x, dtype=torch.long)

# --- DataModule -------------------------------------------------------------

class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        block_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 2,
        download_url: str = None,
        # HF BPE controls
        bpe_vocab_size: int = 1024,
        bpe_min_frequency: int = 2,
        bpe_lowercase: bool = True,
        bpe_special_tokens: Optional[list] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download_url = download_url

        # HF BPE settings
        self.bpe_vocab_size = int(bpe_vocab_size)
        self.bpe_min_frequency = int(bpe_min_frequency)
        self.bpe_lowercase = bool(bpe_lowercase)
        self.bpe_special_tokens = bpe_special_tokens if bpe_special_tokens is not None else ["<unk>", "<pad>", "<bos>", "<eos>"]
        self._hf_bpe_file = None  # will be set when trained/located

        self.tokenizer = None
        self.vocab_size = None

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "tiny_shakespeare.txt")

        # 1) Download dataset if missing
        if not os.path.exists(file_path):
            assert self.download_url is not None, "Provide a download_url for Tiny Shakespeare."
            r = requests.get(self.download_url)
            r.raise_for_status()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(r.text)

        # 2) Ensure a HF BPE tokenizer is trained and saved
        self._hf_bpe_file = os.path.join(self.data_dir, f"hf_bpe_{self.bpe_vocab_size}.json")
        if not os.path.exists(self._hf_bpe_file):
            try:
                from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers  # type: ignore
            except Exception as e:
                raise ImportError("Please `pip install tokenizers` to use tokenizer_type=hf_bpe.") from e

            # Build BPE tokenizer
            tok = Tokenizer(models.BPE(unk_token="<unk>"))

            norm_steps = [normalizers.NFKC()]
            if self.bpe_lowercase:
                norm_steps.append(normalizers.Lowercase())
            tok.normalizer = normalizers.Sequence(norm_steps)

            # ByteLevel is robust for raw text; add_prefix_space=True helps word-boundary merges
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

            trainer = trainers.BpeTrainer(
                vocab_size=self.bpe_vocab_size,
                min_frequency=self.bpe_min_frequency,
                special_tokens=list(self.bpe_special_tokens),
                show_progress=True,
            )

            tok.train(files=[file_path], trainer=trainer)
            tok.save(self._hf_bpe_file)

    def _load_text(self) -> str:
        file_path = os.path.join(self.data_dir, "tiny_shakespeare.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_tokenizer(self, text: str):
        # Only supported tokenizer: HF BPE
        if self._hf_bpe_file is None:
            self._hf_bpe_file = os.path.join(self.data_dir, f"hf_bpe_{self.bpe_vocab_size}.json")
        if not os.path.exists(self._hf_bpe_file):
            raise FileNotFoundError(
                f"HuggingFace BPE tokenizer not found at {self._hf_bpe_file}. "
                f"This should have been created in prepare_data; check dependencies."
            )
        return HFTokenizer(self._hf_bpe_file)

    def setup(self, stage: Optional[str] = None):
        text = self._load_text()
        self.tokenizer = self._build_tokenizer(text)
        self.vocab_size = getattr(self.tokenizer, "vocab_size", None)

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