"""
Scaling law experiment runner — fits in 12.5 GPU-h on A100x2.

Validates: monotonic accuracy with context length under STDP-inference.

Compressed protocol (12.5h budget):
  Phase 1 — Gate (3h): {1K, 10K, 100K} × {A baseline, B +STDP-inference}
                       × NIAH UUID, n=30 per cell. 6 cells × 30 = 180 runs.
  Phase 2 — Full (8h): If gate passes (B > A everywhere), expand to
                       {1K, 10K, 100K, 1M} × {A, D=full stack} × {NIAH, passkey}.
                       8 cells × 30 = 240 runs.
  Phase 3 — Reserve (1.5h): re-run failures, generate plots, compute log-fit.

Run:
  SYNAPFORGE_STDP_INFERENCE=on python scripts/run_scaling_law.py \\
      --ckpt /workspace/runs/synapforge_v41_neuromcp/best.pt \\
      --tokenizer-path Qwen/Qwen2.5-0.5B \\
      --out /workspace/runs/scaling_law

Output:
  /workspace/runs/scaling_law/phase1_gate.json
  /workspace/runs/scaling_law/phase2_full.json
  /workspace/runs/scaling_law/curve.png  (matplotlib)
  /workspace/runs/scaling_law/summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List


def _build_generate_fn(ckpt: str, tokenizer_path: str, max_new: int = 80):
    """Build a callable that takes prompt str → completion str."""
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, "/workspace")
    from synapforge.model_chat_600m import SynapForgeChat600M

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    model = SynapForgeChat600M()
    sd = torch.load(ckpt, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()

    def generate(prompt: str) -> str:
        ids = tok.encode(prompt, return_tensors="pt").cuda()
        if ids.shape[1] > 100_000:
            ids = ids[:, -100_000:]
        with torch.no_grad():
            for _ in range(max_new):
                if hasattr(model, "encode") and hasattr(model, "lm_logits"):
                    h = model.encode(ids)
                    logits = model.lm_logits(h)
                else:
                    logits = model(ids)
                next_id = int(logits[0, -1].argmax().item())
                if next_id == tok.eos_token_id:
                    break
                ids = torch.cat([ids, torch.tensor([[next_id]], device="cuda")], dim=-1)
        completion_ids = ids[0, -max_new:].tolist()
        return tok.decode(completion_ids, skip_special_tokens=True)

    return generate, model


def reset_stdp(model) -> None:
    """Clear STDP plasticity buffers between examples (per-doc consolidation)."""
    import torch

    for module in model.modules():
        if hasattr(module, "reset_doc_state"):
            module.reset_doc_state()
        elif hasattr(module, "W") and hasattr(module, "pre_trace"):
            with torch.no_grad():
                module.W.zero_()
                if hasattr(module, "pre_trace"):
                    module.pre_trace.zero_()
                if hasattr(module, "post_trace"):
                    module.post_trace.zero_()


def run_phase(
    gen_fn: Callable[[str], str],
    model,
    label: str,
    stdp_mode: str,
    context_lens: List[int],
    tasks: List[str],
    n_per_cell: int,
    out_dir: Path,
    reset_between_examples: bool = True,
) -> dict:
    """Run one (label, stdp_mode) combination across all (length, task) cells."""
    from synapforge.eval.niah import run_niah

    os.environ["SYNAPFORGE_STDP_INFERENCE"] = stdp_mode

    results = {
        "label": label,
        "stdp_mode": stdp_mode,
        "tasks": {},
    }

    for task in tasks:
        print(f"=== {label} (stdp={stdp_mode}) task={task} ===", flush=True)

        if reset_between_examples:
            wrapped_gen = gen_fn
            def reset_aware_gen(prompt):
                reset_stdp(model)
                return wrapped_gen(prompt)
            actual_gen = reset_aware_gen
        else:
            actual_gen = gen_fn

        summary = run_niah(
            generate_fn=actual_gen,
            context_lens=context_lens,
            n_per_length=n_per_cell,
            task_kind=task,
            out_path=str(out_dir / f"{label}_{stdp_mode}_{task}.json"),
        )
        results["tasks"][task] = summary
        print(f"  per_len pass: {summary['per_length']}", flush=True)

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer-path", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out", default="/workspace/runs/scaling_law")
    p.add_argument("--phase", default="auto", choices=["auto", "1", "2", "smoke"])
    p.add_argument("--n-gate", type=int, default=30)
    p.add_argument("--n-full", type=int, default=30)
    p.add_argument("--max-new", type=int, default=80)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "smoke":
        print("loading model...")
        gen_fn, model = _build_generate_fn(args.ckpt, args.tokenizer_path, max_new=args.max_new)
        print(f"smoke: 3 examples at ctx=100")
        run_phase(gen_fn, model, "smoke", "on", [100], ["uuid"], 3, out_dir)
        print("smoke OK")
        return

    print(f"=== loading model from {args.ckpt} ===", flush=True)
    gen_fn, model = _build_generate_fn(args.ckpt, args.tokenizer_path, max_new=args.max_new)

    if args.phase in ("auto", "1"):
        print("\n=== Phase 1 — Gate experiment ===", flush=True)
        gate_lens = [1_000, 10_000, 100_000]
        gate_tasks = ["uuid"]

        baseline = run_phase(
            gen_fn, model, label="A_baseline", stdp_mode="off",
            context_lens=gate_lens, tasks=gate_tasks,
            n_per_cell=args.n_gate, out_dir=out_dir,
        )
        with_stdp = run_phase(
            gen_fn, model, label="B_stdp_on", stdp_mode="on",
            context_lens=gate_lens, tasks=gate_tasks,
            n_per_cell=args.n_gate, out_dir=out_dir,
        )

        gate_pass = True
        for ctx_len in gate_lens:
            base_rate = baseline["tasks"]["uuid"]["per_length"][ctx_len]["pass_rate"]
            stdp_rate = with_stdp["tasks"]["uuid"]["per_length"][ctx_len]["pass_rate"]
            print(f"  ctx={ctx_len}: A={base_rate:.2f} vs B={stdp_rate:.2f} "
                  f"({'PASS' if stdp_rate > base_rate else 'FAIL'})", flush=True)
            if stdp_rate <= base_rate:
                gate_pass = False

        with open(out_dir / "phase1_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "gate_pass": gate_pass,
                "A_baseline": baseline,
                "B_stdp_on": with_stdp,
            }, f, indent=2)

        if not gate_pass:
            print("\n=== GATE FAILED. STDP inference does not dominate baseline. ===")
            print("  Skipping Phase 2. Fall back to engineering improvements only.")
            return
        print("\n=== GATE PASSED. STDP inference dominates baseline at all 3 lengths. ===")

    if args.phase in ("auto", "2"):
        print("\n=== Phase 2 — Full scaling-law experiment ===", flush=True)
        full_lens = [1_000, 10_000, 100_000, 1_000_000]
        full_tasks = ["uuid", "passkey"]

        baseline = run_phase(
            gen_fn, model, label="A_full_baseline", stdp_mode="off",
            context_lens=full_lens, tasks=full_tasks,
            n_per_cell=args.n_full, out_dir=out_dir,
        )
        full_stack = run_phase(
            gen_fn, model, label="D_full_stack", stdp_mode="on",
            context_lens=full_lens, tasks=full_tasks,
            n_per_cell=args.n_full, out_dir=out_dir,
        )

        with open(out_dir / "phase2_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "A_full_baseline": baseline,
                "D_full_stack": full_stack,
            }, f, indent=2)

        print("\n=== Phase 2 complete. ===")
        for ctx_len in full_lens:
            for task in full_tasks:
                base_rate = baseline["tasks"][task]["per_length"][ctx_len]["pass_rate"]
                full_rate = full_stack["tasks"][task]["per_length"][ctx_len]["pass_rate"]
                print(f"  ctx={ctx_len:>10} task={task:8} "
                      f"A={base_rate:.2f} D={full_rate:.2f} "
                      f"delta={full_rate-base_rate:+.2f}")

    print(f"\n=== All phases done. Results in {out_dir}/ ===")


if __name__ == "__main__":
    main()
