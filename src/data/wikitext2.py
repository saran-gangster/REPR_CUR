import os
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

# --- Tokenizer wrapper ------------------------------------------------------

class HFTokenizer:
    def __init__(self, tokenizer_json_path: str):
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as e:
            raise ImportError("Please `pip install tokenizers` to use the HF BPE tokenizer.") from e
        self.tk = Tokenizer.from_file(tokenizer_json_path)
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, s: str) -> List[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: List[int]) -> str:
        try:
            return self.tk.decode(ids)
        except Exception:
            return ""

# --- Datasets ---------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data  # int32 array of token ids
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.data) // self.block_size)

    def __getitem__(self, idx):
        start = np.random.randint(0, len(self.data) - self.block_size - 1)
        x = self.data[start : start + self.block_size]
        return torch.tensor(x, dtype=torch.long)

class EvalDataset(Dataset):
    """
    Deterministic, non-overlapping windows for validation/test to stabilize metrics.
    """
    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data
        self.block_size = block_size
        L = len(self.data)
        self.n = max(1, (L - 1) // block_size)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.data[start : start + self.block_size]
        return torch.tensor(x, dtype=torch.long)

# --- DataModule -------------------------------------------------------------

class WikiText2DataModule(L.LightningDataModule):
    """
    LightningDataModule for WikiText-2:
    - Uses datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    - Trains a custom HF BPE tokenizer on the training split
    - Supports training on a random fraction of the training set via train_fraction
    """
    def __init__(
        self,
        data_dir: str = "data/wikitext2/",
        block_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 2,
        # HF datasets cache (optional)
        hf_cache_dir: Optional[str] = None,
        # BPE settings
        bpe_vocab_size: int = 32000,
        bpe_min_frequency: int = 2,
        bpe_lowercase: bool = True,
        bpe_special_tokens: Optional[List[str]] = None,
        # Fast prototyping: use fraction of train split
        train_fraction: float = 1.0,  # e.g., 0.2 for 20%
        subset_seed: int = 1234,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.hf_cache_dir = hf_cache_dir

        # HF BPE settings
        self.bpe_vocab_size = int(bpe_vocab_size)
        self.bpe_min_frequency = int(bpe_min_frequency)
        self.bpe_lowercase = bool(bpe_lowercase)
        self.bpe_special_tokens = bpe_special_tokens if bpe_special_tokens is not None else ["<unk>", "<pad>", "<bos>", "<eos>"]

        # Fractional training subset
        self.train_fraction = float(train_fraction)
        self.subset_seed = int(subset_seed)

        os.makedirs(self.data_dir, exist_ok=True)
        suffix = "_lower" if self.bpe_lowercase else "_raw"
        self._hf_bpe_file = os.path.join(self.data_dir, f"hf_bpe_wt2_{self.bpe_vocab_size}{suffix}.json")

        self.tokenizer: Optional[HFTokenizer] = None
        self.vocab_size: Optional[int] = None

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def prepare_data(self):
        # Ensure dependencies exist
        try:
            import datasets  # type: ignore
        except Exception as e:
            raise ImportError("Please `pip install datasets` to use WikiText-2.") from e

        # Download dataset (only in rank 0)
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=self.hf_cache_dir)

        # Train and save tokenizer once
        if not os.path.exists(self._hf_bpe_file):
            try:
                from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers  # type: ignore
            except Exception as e:
                raise ImportError("Please `pip install tokenizers` to train the HF BPE tokenizer.") from e

            # Build training text file to follow our existing pattern
            train_texts = ds["train"]["text"]
            train_texts = [t if t is not None else "" for t in train_texts]
            train_text = "\n".join(train_texts)
            raw_train_file = os.path.join(self.data_dir, "wikitext2_train_raw.txt")
            with open(raw_train_file, "w", encoding="utf-8") as f:
                f.write(train_text)

            # Build BPE tokenizer
            tok = Tokenizer(models.BPE(unk_token="<unk>"))
            norm_steps = [normalizers.NFKC()]
            if self.bpe_lowercase:
                norm_steps.append(normalizers.Lowercase())
            tok.normalizer = normalizers.Sequence(norm_steps)
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

            trainer = trainers.BpeTrainer(
                vocab_size=self.bpe_vocab_size,
                min_frequency=self.bpe_min_frequency,
                special_tokens=list(self.bpe_special_tokens),
                show_progress=True,
            )

            tok.train(files=[raw_train_file], trainer=trainer)
            tok.save(self._hf_bpe_file)

    def _load_dataset_splits(self):
        import datasets  # type: ignore
        return datasets.load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=self.hf_cache_dir)

    def _build_tokenizer(self) -> HFTokenizer:
        if not os.path.exists(self._hf_bpe_file):
            raise FileNotFoundError(
                f"HuggingFace BPE tokenizer not found at {self._hf_bpe_file}. "
                f"It should have been created in prepare_data; check dependencies."
            )
        return HFTokenizer(self._hf_bpe_file)

    def _subset_train_texts(self, texts: List[str]) -> List[str]:
        if self.train_fraction >= 1.0:
            return texts
        n = len(texts)
        k = max(1, int(n * self.train_fraction))
        rng = np.random.default_rng(self.subset_seed)
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        return [texts[i] for i in idx]

    def setup(self, stage: Optional[str] = None):
        # Tokenizer
        self.tokenizer = self._build_tokenizer()
        self.vocab_size = getattr(self.tokenizer, "vocab_size", None)

        # Load dataset splits
        ds = self._load_dataset_splits()

        # Train split (optionally subset for fast prototyping)
        train_texts = [t if t is not None else "" for t in ds["train"]["text"]]
        train_texts = self._subset_train_texts(train_texts)
        train_text = "\n".join(train_texts)

        # Validation split (deterministic windows)
        val_texts = [t if t is not None else "" for t in ds["validation"]["text"]]
        val_text = "\n".join(val_texts)

        # Test split (optional; mirrors validation handling)
        test_texts = [t if t is not None else "" for t in ds.get("test", {"text": []})["text"]]
        test_text = "\n".join(test_texts)

        # Encode to ids
        train_ids = np.array(self.tokenizer.encode(train_text), dtype=np.int32)
        val_ids = np.array(self.tokenizer.encode(val_text), dtype=np.int32)
        test_ids = np.array(self.tokenizer.encode(test_text), dtype=np.int32) if len(test_text) > 0 else np.zeros((1,), dtype=np.int32)

        # Datasets
        self.train_ds = CharDataset(train_ids, self.block_size)
        self.val_ds = EvalDataset(val_ids, self.block_size)
        self.test_ds = EvalDataset(test_ids, self.block_size) if len(test_ids) > 1 else None

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

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
        if self.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(self.test_ds, **kwargs)