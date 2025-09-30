from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from typing import Dict, Type

import lightning as L


def _snake_case(s: str) -> str:
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _available_datamodule_classes() -> Dict[str, Type[L.LightningDataModule]]:
    """
    Discover all LightningDataModule subclasses in src.data.* submodules.
    Returns a mapping: alias -> class
    """
    try:
        import src.data as data_pkg  # noqa: F401
    except Exception as e:
        raise ImportError("Cannot import package src.data. Ensure src is on PYTHONPATH.") from e

    registry: Dict[str, Type[L.LightningDataModule]] = {}
    # Iterate over modules in src.data package
    for modinfo in pkgutil.iter_modules(data_pkg.__path__):
        mod_name = modinfo.name
        if mod_name.startswith("_") or mod_name in {"registry", "unified", "__init__", "tokenization", "hf_tokenizer"}:
            continue
        try:
            module = importlib.import_module(f"src.data.{mod_name}")
        except Exception:
            # Skip modules that fail to import here; their deps may be optional
            continue

        # Find all LightningDataModule classes in the module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if issubclass(obj, L.LightningDataModule):
                    # Build aliases from module name and class name
                    aliases = set()

                    # Module name (e.g., "wikitext2")
                    aliases.add(_snake_case(mod_name))

                    # Class name variants (e.g., "WikiText2DataModule" -> "wiki_text2")
                    clsname = obj.__name__
                    base = clsname.replace("DataModule", "")
                    aliases.add(_snake_case(base))
                    aliases.add(_snake_case(clsname))

                    # Heuristics for common names
                    lowered = {a.replace("_", "") for a in list(aliases)}
                    aliases |= lowered

                    # Allow classes to define extra aliases via class attribute ALIASES
                    extra = getattr(obj, "ALIASES", None)
                    if extra:
                        for a in extra:
                            aliases.add(_snake_case(a))
                            aliases.add(_snake_case(a).replace("_", ""))

                    # Special-cases for convenience
                    if "wikitext2" in aliases or "wikitext" in aliases or "wiki" in aliases:
                        aliases.update({"wikitext2", "wikitext", "wiki", "wt2"})
                    if "tinyshakespeare" in aliases or "tiny_shakespeare" in aliases or "shakespeare" in aliases:
                        aliases.update({"tiny", "tinyshakespeare", "tiny_shakespeare", "shakespeare"})

                    for a in aliases:
                        registry.setdefault(a, obj)
            except Exception:
                continue
    return registry


_REGISTRY: Dict[str, Type[L.LightningDataModule]] | None = None


def _get_registry() -> Dict[str, Type[L.LightningDataModule]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _available_datamodule_classes()
    return _REGISTRY


def resolve_datamodule_class(name_or_path: str) -> Type[L.LightningDataModule]:
    """
    Resolve a DataModule class from:
      - shorthand alias (e.g., "wikitext2", "tiny_shakespeare", "wiki", "wt2", "tiny", "shakespeare")
      - module path string "src.data.module:ClassName" or "package.module:ClassName"
      - bare module name "my_dataset" -> src.data.my_dataset (first LightningDataModule in that module)
    """
    s = (name_or_path or "").strip()
    if not s:
        raise ValueError("Empty data.module provided. Expected an alias (e.g., 'wikitext2') or a class path 'pkg.mod:Class'.")

    # Full path form: "pkg.module:ClassName"
    if ":" in s:
        mod_str, cls_name = s.split(":", 1)
        try:
            mod = importlib.import_module(mod_str)
            cls = getattr(mod, cls_name)
            if not issubclass(cls, L.LightningDataModule):
                raise TypeError(f"{s} is not a LightningDataModule (got {cls}).")
            return cls
        except Exception as e:
            raise ImportError(f"Could not import DataModule from '{s}'. Error: {e}") from e

    # Alias shortcut
    reg = _get_registry()
    key = _snake_case(s)
    key_nosymbol = key.replace("_", "")
    if key in reg:
        return reg[key]
    if key_nosymbol in reg:
        return reg[key_nosymbol]

    # Try importing src.data.<s> and pick first LightningDataModule found
    try:
        mod = importlib.import_module(f"src.data.{_snake_case(s)}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, L.LightningDataModule):
                return obj
    except Exception:
        pass

    # Helpful error
    available = sorted(set(_get_registry().keys()))
    raise ValueError(
        f"Could not resolve data.module='{s}'.\n"
        f"Try one of: {available}\n"
        f"Or specify a full class path like 'src.data.wikitext2:WikiText2DataModule'."
    )


def available_datamodules() -> Dict[str, str]:
    reg = _get_registry()
    return {k: f"{v.__module__}:{v.__qualname__}" for k, v in sorted(reg.items())}