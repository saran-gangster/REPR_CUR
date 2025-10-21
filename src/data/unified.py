from __future__ import annotations
from typing import Any, Dict, Optional
import lightning as L
from .registry import resolve_datamodule_class

class UnifiedDataModule(L.LightningDataModule):
    """
    A wrapper that dynamically instantiates a concrete LightningDataModule at runtime.

    Usage:
      - Set data.module to:
          - an alias like "wikitext2" or "tiny_shakespeare", OR
          - a full path "src.data.wikitext2:WikiText2DataModule".
      - Any other kwargs in the 'data:' config are forwarded to the underlying DataModule.
    """

    def __init__(
        self, 
        module: str = "wikitext2",
        # Explicitly declare common parameters to satisfy Lightning CLI validation
        data_dir: Optional[str] = None,
        block_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        download_url: Optional[str] = None,
        bpe_vocab_size: Optional[int] = None,
        bpe_min_frequency: Optional[int] = None,
        bpe_lowercase: Optional[bool] = None,
        bpe_special_tokens: Optional[list] = None,
        hf_cache_dir: Optional[str] = None,
        train_fraction: Optional[float] = None,
        subset_seed: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__()
        self._module_id: str = str(module)
        
        # Collect all provided arguments (both explicit and kwargs)
        self._dm_kwargs: Dict[str, Any] = dict(kwargs)

        optional_args = {
            "data_dir": data_dir,
            "block_size": block_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "download_url": download_url,
            "bpe_vocab_size": bpe_vocab_size,
            "bpe_min_frequency": bpe_min_frequency,
            "bpe_lowercase": bpe_lowercase,
            "bpe_special_tokens": bpe_special_tokens,
            "hf_cache_dir": hf_cache_dir,
            "train_fraction": train_fraction,
            "subset_seed": subset_seed,
        }
        self._dm_kwargs.update({k: v for k, v in optional_args.items() if v is not None})
            
        self._dm: Optional[L.LightningDataModule] = None

        # Common passthrough attributes that some code expects (e.g., model auto-vocab sizing)
        self.vocab_size: Optional[int] = None
        self.tokenizer: Any = None

    # Internal: instantiate the underlying DataModule if needed
    def _ensure_dm(self):
        if self._dm is None:
            cls = resolve_datamodule_class(self._module_id)
            # Filter kwargs to only pass those that the target class accepts
            import inspect
            sig = inspect.signature(cls.__init__)
            valid_kwargs = {}
            for k, v in self._dm_kwargs.items():
                if k in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD 
                    for p in sig.parameters.values()
                ):
                    valid_kwargs[k] = v
            self._dm = cls(**valid_kwargs)

    # Lightning hooks
    def prepare_data(self):
        self._ensure_dm()
        if hasattr(self._dm, "prepare_data"):
            self._dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self._ensure_dm()
        if hasattr(self._dm, "setup"):
            self._dm.setup(stage=stage)
        # propagate commonly used attributes
        self.vocab_size = getattr(self._dm, "vocab_size", None)
        self.tokenizer = getattr(self._dm, "tokenizer", None)

    def teardown(self, stage: Optional[str] = None):
        if self._dm is not None and hasattr(self._dm, "teardown"):
            self._dm.teardown(stage=stage)

    # Dataloaders: delegate to underlying
    def train_dataloader(self):
        self._ensure_dm()
        return self._dm.train_dataloader()

    def val_dataloader(self):
        self._ensure_dm()
        return self._dm.val_dataloader()

    def test_dataloader(self):
        self._ensure_dm()
        if hasattr(self._dm, "test_dataloader"):
            return self._dm.test_dataloader()
        return None

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(module={self._module_id!r}, kwargs={list(self._dm_kwargs.keys())})"