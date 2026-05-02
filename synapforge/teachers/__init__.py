"""synapforge.teachers — frozen KD teacher loaders.

Exists to centralise teacher-side machinery away from
``train_100m_kd.py`` so each teacher backend (GPU fp16, GPU bf16, CPU
fp32, CPU INT8) can ship with its own load + forward semantics and a
unit test next to it.

Currently shipped
-----------------
* ``cpu_int8`` — load any HF CausalLM on CPU with optional
  bitsandbytes INT8 quantisation. Designed to free the ~2 GB of HBM
  that the Qwen 2.5 0.5B teacher would otherwise consume on the
  rental A800.

Roadmap
-------
* ``gpu_bf16`` — promote the in-line GPU loader from train_100m_kd.py
  to its own module once the second backend lands (this one).
"""
from __future__ import annotations
