"""Tests for sf.hf_trainer — verify HuggingFace Trainer integration.

50-step training run with a tiny sf.Module (sf.HybridBlock + LM head).
We ship our own tiny tokenizer + dataset so the test does not need
internet access; in production, swap in
``sf.huggingface_adapter.load_tokenizer + load_dataset_streaming``.

Run: CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python -m pytest \\
        /workspace/synapforge/test_hf_trainer.py -v -s
or as script: /opt/conda/bin/python /workspace/synapforge/test_hf_trainer.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
import synapforge as sf
from synapforge.hf_trainer import SFTrainer
from synapforge.plasticity import Hebbian, PlasticityEngine

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Tiny LM model (sf.HybridBlock body + tied LM head)
# ---------------------------------------------------------------------------


class TinyHybridLM(sf.Module):
    """Embed -> sf.LiquidCell -> sf.PLIF -> Linear LM head -> token logits.

    Returns a HuggingFace-style ``ModelOutput`` so the parent ``Trainer.compute_loss``
    can find ``.loss`` directly.
    """

    def __init__(self, vocab: int = 64, d: int = 32) -> None:
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.cfc = sf.LiquidCell(d, d)
        self.plif = sf.PLIF(d, threshold=0.3)
        self.proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_unused,
    ):
        # input_ids: (B, T)
        x = self.embed(input_ids)
        h = self.cfc(x)
        spk, _mem = self.plif(h)
        proj = self.proj(spk)
        # Use the post-projection signal blended with a residual from h so
        # gradients flow through PLIF's surrogate AND a non-spiking path.
        feat = proj + h
        logits = self.lm_head(feat)
        loss = None
        if labels is not None:
            # Causal LM: predict t+1 from t.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab),
                shift_labels.view(-1),
            )
        # Return a dict-like ModelOutput so HF Trainer.compute_loss can
        # subscript with ``outputs["loss"]`` and unpack ``**outputs``.
        from transformers.modeling_outputs import ModelOutput
        return ModelOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Tiny dataset — char-level repeating pattern so the model CAN actually fit
# ---------------------------------------------------------------------------


class RepeatingPatternDataset(torch.utils.data.Dataset):
    """Each sample is a 32-token repeating pattern: a permutation cycle.

    The model's job is to predict the next token in the cycle. With 50 steps
    the loss should fall noticeably (from ln(vocab) baseline to <2.5).
    """

    def __init__(self, n: int = 256, seq_len: int = 32, vocab: int = 64) -> None:
        self.n = int(n)
        self.seq_len = int(seq_len)
        self.vocab = int(vocab)
        rng = torch.Generator().manual_seed(123)
        self.cycles: List[torch.Tensor] = []
        for _ in range(self.n):
            # Each row is a fixed-length permutation/repeat over 8 unique tokens.
            base = torch.randperm(8, generator=rng) % vocab
            ids = base.repeat((seq_len + 7) // 8)[:seq_len]
            self.cycles.append(ids.long())

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.cycles[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def collate(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


# ---------------------------------------------------------------------------
# Test 1: 50-step trainer.train(), verify loss drops + no NaN
# ---------------------------------------------------------------------------


def test_sf_trainer_50_steps_loss_decreases():
    from transformers import TrainingArguments, TrainerCallback

    # Disable accelerate's distributed init helpers (single-GPU); HF still
    # uses Accelerate internally for autocast/data movement.
    os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
    os.environ.setdefault("ACCELERATE_USE_FSDP", "false")

    vocab, d = 64, 32
    model = TinyHybridLM(vocab=vocab, d=d).to(DEV)
    ds = RepeatingPatternDataset(n=256, seq_len=32, vocab=vocab)

    out_dir = "/tmp/sf_hf_trainer_smoke"
    os.makedirs(out_dir, exist_ok=True)

    losses_collected: List[float] = []

    class _Capture(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses_collected.append(float(logs["loss"]))

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,
        max_steps=50,
        learning_rate=3e-3,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        bf16=True,
        seed=0,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
    )
    trainer = SFTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        callbacks=[_Capture()],
    )

    t0 = time.perf_counter()
    out = trainer.train()
    elapsed = time.perf_counter() - t0

    print(f"\n[hf_trainer test] 50 steps in {elapsed*1000:.0f} ms "
          f"({elapsed*1000/50:.1f} ms/step)")
    print(f"[hf_trainer test] losses logged: {losses_collected}")
    print(f"[hf_trainer test] final train loss (state.log_history): "
          f"{trainer.state.log_history[-1] if trainer.state.log_history else 'NA'}")

    # Loss must be finite throughout.
    for l in losses_collected:
        assert math.isfinite(l), f"non-finite loss observed: {l}"

    # Loss must DECREASE: first vs last logged loss.
    assert len(losses_collected) >= 2, "need >= 2 loss points to compare"
    first, last = losses_collected[0], losses_collected[-1]
    assert last < first, f"loss did not decrease: first={first:.3f} last={last:.3f}"
    # Stronger: at least 5% drop relative.
    rel_drop = (first - last) / max(first, 1e-9)
    assert rel_drop > 0.05, (
        f"loss drop too small: first={first:.3f} last={last:.3f} "
        f"rel={rel_drop:.3f}"
    )
    print(f"[hf_trainer test] loss {first:.3f} -> {last:.3f} "
          f"({rel_drop*100:.1f}% drop)")


# ---------------------------------------------------------------------------
# Test 2: SFTrainer with a plasticity_engine wired in
# ---------------------------------------------------------------------------


def test_sf_trainer_with_plasticity_engine():
    """Hebbian rule observes during forward; engine fires after each opt.step."""
    from transformers import TrainingArguments

    vocab, d = 32, 16
    model = TinyHybridLM(vocab=vocab, d=d).to(DEV)
    ds = RepeatingPatternDataset(n=64, seq_len=16, vocab=vocab)

    # Build a Hebbian rule that targets the LM head's weight as its observable.
    hebb = Hebbian(lr=1e-4).to(DEV)

    def _observe(module, inputs, outputs):
        # Hook into the proj layer: pre = inputs[0] (post-PLIF feature),
        # post = outputs (proj output). Register on the .proj layer below.
        pre = inputs[0].detach().float()
        post = outputs.detach().float()
        hebb.observe(pre=pre, post=post, t=0.0)

    handle = model.proj.register_forward_hook(_observe)
    engine = PlasticityEngine(
        rules={"proj.weight": hebb},
        schedule="every:5",
    )

    out_dir = "/tmp/sf_hf_trainer_plast"
    os.makedirs(out_dir, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        max_steps=20,
        learning_rate=1e-3,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = SFTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        plasticity_engine=engine,
    )
    trainer.train()
    handle.remove()
    # The plasticity callback should have applied something at least once
    # under "every:5" schedule across 20 steps -> 4 applies.
    cb = trainer._plasticity_callback
    assert cb is not None
    print(f"[hf_trainer plast] engine.apply called {cb._n_applied} times")
    assert cb._n_applied >= 1, "expected at least one plasticity apply call"


# ---------------------------------------------------------------------------
# Test 3: bf16 mixed-precision path runs without dtype conflict
# ---------------------------------------------------------------------------


def test_sf_trainer_bf16_runs():
    """bf16=True must not crash even though sf.LiquidCell promotes inside scan."""
    from transformers import TrainingArguments

    vocab, d = 32, 16
    model = TinyHybridLM(vocab=vocab, d=d).to(DEV)
    ds = RepeatingPatternDataset(n=32, seq_len=16, vocab=vocab)
    args = TrainingArguments(
        output_dir="/tmp/sf_hf_trainer_bf16",
        per_device_train_batch_size=4,
        max_steps=5,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
        seed=0,
    )
    trainer = SFTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()
    print("[hf_trainer bf16] passed.")


if __name__ == "__main__":
    failures = 0
    for name in [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]:
        try:
            print(f"\n=== {name} ===", flush=True)
            globals()[name]()
            print(f"  PASS", flush=True)
        except Exception as exc:
            print(f"  FAIL: {exc!r}", flush=True)
            import traceback
            traceback.print_exc()
            failures += 1
    print(f"\nSummary: {failures} failure(s)", flush=True)
    sys.exit(1 if failures else 0)
