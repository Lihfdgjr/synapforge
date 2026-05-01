#!/usr/bin/env bash
# Reproduce every number in docs/RFOLD_PAPER.md.
#
# Idempotent: re-running overwrites paper_repro/*.json. No global state.
# Output: paper_repro/{rfold_correctness.json, rfold_speed_cpu.json, rfold_speed_gpu.json}
#
# Usage:
#     bash scripts/rfold_paper_repro.sh
#
# Requirements:
#     - Python with torch installed (CPU is enough; GPU enables rfold_speed_gpu.json)
#     - This script must be run from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="$REPO_ROOT/paper_repro"
mkdir -p "$OUT"

PYTHON="${PYTHON:-python}"

echo "[rfold-repro] repo root: $REPO_ROOT"
echo "[rfold-repro] output dir: $OUT"
echo "[rfold-repro] python: $($PYTHON --version 2>&1)"
echo

# 1. Correctness via verify_rfold.py (asserts internally)
echo "[rfold-repro] (1/3) correctness: scripts/verify_rfold.py"
"$PYTHON" scripts/verify_rfold.py | tee "$OUT/_verify_rfold.log"

# 2. Programmatic capture of correctness + ablation as JSON
echo "[rfold-repro] (2/3) correctness JSON  ->  $OUT/rfold_correctness.json"
"$PYTHON" - <<'PY' | tee "$OUT/rfold_correctness.json"
import json, math, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent if False else "."))
from synapforge.cells.rfold import cfc_rfold, cfc_rfold_chunked, _sequential_cfc

def rand(B, N, D, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, N, generator=g) * 0.1,
        torch.randn(B, D, generator=g) * 0.5,
        torch.randn(N, D, generator=g) * (scale / math.sqrt(D)),
        torch.randn(N, N, generator=g) * (scale / math.sqrt(N)),
        torch.randn(N, N, generator=g) * (scale / math.sqrt(N)),
        torch.randn(N, generator=g) * 0.5 - 1.0,
    )

results = {"correctness": {}, "ablation_chunk": []}
h0, x, Wi, Wh, Wg, tau = rand(4, 16, 8)
for R in [1, 8, 16]:
    h_seq  = _sequential_cfc(h0, x, Wi, Wh, Wg, tau, R)
    h_fold = cfc_rfold(h0, x, Wi, Wh, Wg, tau, R)
    err = ((h_seq - h_fold).norm() / (h_seq.norm() + 1e-12)).item()
    results["correctness"][f"R={R}"] = err

# Chunk ablation at R=8
h0b, xb, Wib, Whb, Wgb, taub = rand(4, 24, 12, scale=0.4)
h_seq  = _sequential_cfc(h0b, xb, Wib, Whb, Wgb, taub, 8)
for chunk in [8, 4, 2, 1]:
    if chunk == 1:
        h_c = h_seq
    elif chunk == 8:
        h_c = cfc_rfold(h0b, xb, Wib, Whb, Wgb, taub, 8)
    else:
        h_c = cfc_rfold_chunked(h0b, xb, Wib, Whb, Wgb, taub, 8, chunk=chunk)
    err = (h_seq - h_c).norm().item()
    results["ablation_chunk"].append({"chunk": chunk, "abs_err": err})

print(json.dumps(results, indent=2))
PY

# 3. Speed sweep on CPU and (if available) GPU
echo "[rfold-repro] (3/3) speed JSON      ->  $OUT/rfold_speed_cpu.json + rfold_speed_gpu.json"
"$PYTHON" - <<'PY'
import json, math, time, torch
from pathlib import Path
from synapforge.cells.rfold import cfc_rfold, _sequential_cfc

def rand(B, N, D, seed=0, scale=0.3, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    out = (
        torch.randn(B, N, generator=g) * 0.1,
        torch.randn(B, D, generator=g) * 0.5,
        torch.randn(N, D, generator=g) * (scale / math.sqrt(D)),
        torch.randn(N, N, generator=g) * (scale / math.sqrt(N)),
        torch.randn(N, N, generator=g) * (scale / math.sqrt(N)),
        torch.randn(N, generator=g) * 0.5 - 1.0,
    )
    return tuple(t.to(device) for t in out)

def bench_device(device):
    rows = []
    use_cuda = (device == "cuda")
    shapes = [(64,4),(64,8),(64,16),(128,8),(256,8),(512,8)]
    for N, R in shapes:
        h0, x, Wi, Wh, Wg, tau = rand(8, N, max(N // 2, 16), device=device)
        # warmup
        _ = _sequential_cfc(h0, x, Wi, Wh, Wg, tau, R)
        _ = cfc_rfold(h0, x, Wi, Wh, Wg, tau, R)
        if use_cuda:
            torch.cuda.synchronize()
        iters = max(10, 80 // max(N // 64, 1))

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = _sequential_cfc(h0, x, Wi, Wh, Wg, tau, R)
        if use_cuda:
            torch.cuda.synchronize()
        t_seq = (time.perf_counter() - t0) / iters * 1000

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = cfc_rfold(h0, x, Wi, Wh, Wg, tau, R)
        if use_cuda:
            torch.cuda.synchronize()
        t_fold = (time.perf_counter() - t0) / iters * 1000
        rows.append({
            "N": N, "R": R,
            "seq_ms": round(t_seq, 4),
            "fold_ms": round(t_fold, 4),
            "speedup": round(t_seq / max(t_fold, 1e-9), 3),
        })
    return rows

out = Path("paper_repro")
out.mkdir(exist_ok=True)
cpu = bench_device("cpu")
(out / "rfold_speed_cpu.json").write_text(json.dumps({"device": "cpu", "rows": cpu}, indent=2))
print("[rfold-repro] CPU rows:", len(cpu))
if torch.cuda.is_available():
    gpu = bench_device("cuda")
    (out / "rfold_speed_gpu.json").write_text(json.dumps({"device": "cuda", "rows": gpu}, indent=2))
    print("[rfold-repro] GPU rows:", len(gpu))
else:
    (out / "rfold_speed_gpu.json").write_text(json.dumps({"device": "cuda", "skipped": "no CUDA available"}, indent=2))
    print("[rfold-repro] CUDA not available; wrote skipped marker.")
PY

echo
echo "[rfold-repro] done."
ls -la "$OUT"
