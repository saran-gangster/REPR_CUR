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

    def __init__(self, module: str = "wikitext2", **kwargs: Any):
        super().__init__()
        self._module_id: str = str(module)
        # Store kwargs; forward them to the underlying datamodule when constructed
        self._dm_kwargs: Dict[str, Any] = dict(kwargs)
        self._dm: Optional[L.LightningDataModule] = None

        # Common passthrough attributes that some code expects (e.g., model auto-vocab sizing)
        self.vocab_size: Optional[int] = None
        self.tokenizer: Any = None

    # Internal: instantiate the underlying DataModule if needed
    def _ensure_dm(self):
        if self._dm is None:
            cls = resolve_datamodule_class(self._module_id)
            self._dm = cls(**self._dm_kwargs)

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

    # Nice to have: readable repr
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(module={self._module_id!r}, kwargs={list(self._dm_kwargs.keys())})"