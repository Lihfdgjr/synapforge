"""synapforge.native -- pure-numpy native kernels and gradient catalogues.

Sub-packages
------------
data/
    Torch-free data loaders (parquet/jsonl) emitting numpy/cupy arrays.
    Replaces ``synapforge.data.ParquetTokenStream`` for the native trainer.

This package is the long-term replacement for torch.autograd in the
SynapForge training and inference paths. Pure numpy (+ optional cupy /
sentencepiece / tokenizers / transformers) front; zero torch in any
``synapforge/native/data/*.py`` module.
"""

from __future__ import annotations

__all__: list[str] = []
