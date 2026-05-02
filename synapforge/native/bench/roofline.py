"""roofline.py -- physical roofline for SynapForge HybridBlock training.

Goal
----
Given (model, hardware), compute the *upper bound* tok/s reachable by an
ideal kernel: the speed at which the HW physically saturates either its
Tensor Core compute, its HBM bandwidth, or its PCIe link. Everything
slower than this number is "kernel slack", not a hardware limit.

Why this file exists (USER 2026-05-02)
--------------------------------------
Run 7 hits ~2750 tok/s on A800 80GB. Theoretical TC roofline is
~30-40k tok/s. We are at <10% utilization. We need a data-driven
breakdown of *which* roofline is binding (compute, HBM, PCIe) before we
can spend engineering on Triton fusion vs. data-loader vs. CPU offload.

Op accounting for HybridBlock (d=1280, ffn_ratio=3.0 -> h=3840)
---------------------------------------------------------------
Per token, per layer, **forward**:

    ln1            : RMSNorm  ~ 8d FLOPs  (negligible vs GEMM)
    cfc_step       : 2 d^2 FLOPs (delta_proj + b_proj)
    plif_step      : ~6d FLOPs (membrane potential update; surrogate is
                                cheap; negligible vs GEMM)
    ln2            : RMSNorm  ~ 8d FLOPs
    swiglu (3 GEMMs):
        gate       : 2 * d * h
        up         : 2 * d * h
        down       : 2 * h * d
                   = 6 * d * h
    sew_shortcut   : 2 d^2 (optional, run7 enables it)

    fwd_per_layer  ~= 6*d*h + 4*d^2 + small
                   = 6*1280*3840 + 4*1280^2
                   = 29_491_200 + 6_553_600
                   = 36_044_800 FLOPs per token per layer

LM head fwd     : 2 * d * V  ~ 2*1280*151936 = 388_956_160 FLOPs/token
embed lookup    : O(d)  (negligible -- gather, no matmul)

Backward is ~2x forward GEMM cost (the "2/3 + 1/3" rule for x_grad +
W_grad). So total step (one fwd + one bwd) per token, per layer:

    step_per_layer ~= 3 * fwd_per_layer = 108_134_400 FLOPs

For n_layers=16 and loop_depth=2 (Run 7 has --loop-depth 2):

    total_layer_step = n_layers * loop_depth * step_per_layer
                     = 16 * 2 * 108_134_400
                     = 3_460_300_800 FLOPs/token

LM head bwd: 2 * fwd cost (one for grad_x, one for grad_W) -> 3 *
388_956_160 = 1_166_868_480 FLOPs/token.

KD-distill (--kd-every 4 --kd-topk 2048 --kd-weight 0.7 in Run 7) adds
small softmax + KL:
    kd_per_token   = 4 * V * 2 (sm + KL fwd+bwd) / kd_every
                   ~= 4 * 151936 * 2 / 4 = 303_872 FLOPs/token
                   negligible.

Total step FLOPs/token ~= 3.46e9 + 1.17e9 = 4.63e9 FLOPs/tok

For B=32 grad_accum=2 T=1024:
    tokens_per_step = 32 * 2 * 1024 = 65_536 tokens
    flops_per_step  = 4.63e9 * 65_536 = 3.04e14 FLOPs = 304 TFLOP

A800 80GB (TF32 312 TFLOPS, bf16 also 312 TFLOPS):
    compute_bound_step_ms = 304e12 / 312e12 * 1000 = 974 ms/step
    -> 65_536 / 0.974 = 67_287 tok/s (TC roof)

But Run 7 uses bf16-full + cpu-offload-optim + grad-ckpt + rfold-chunk,
so HBM and PCIe can each crush this number. The roofline below
computes all three regimes per-stage so we know which one is binding.

References
----------
Williams et al. 2009 "Roofline" CACM: AI-vs-FLOPS triangle.
NVIDIA A800 spec sheet: 312 TFLOPS bf16, 1.555 TB/s HBM2e (1.5 TB/s).
NVIDIA A100 80GB spec: 312 TFLOPS bf16, 2.04 TB/s HBM2e.
NVIDIA H100 80GB SXM: 989 TFLOPS bf16 (no sparsity), 3.35 TB/s HBM3.
PCIe Gen4 x16: 32 GB/s nominal one-way, ~25 GB/s sustained.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Hardware specs -- bf16 TC, HBM, PCIe.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareSpec:
    """Static physical limits of a single GPU + its host link.

    Numbers are from NVIDIA spec sheets. ``tc_tflops`` is "achievable
    bf16 dense Tensor Core TFLOPS" (no 2x sparsity multiplier applied).
    ``hbm_bw_gbs`` is sustained HBM bandwidth in GB/s. ``pcie_bw_gbs``
    is sustained host<->device PCIe bandwidth one-way (real, not
    nominal).
    """

    name: str
    tc_tflops: float
    hbm_bw_gbs: float
    pcie_bw_gbs: float
    sm_count: int
    mem_gb: float

    @property
    def ridge_ai(self) -> float:
        """Arithmetic intensity (FLOPs/byte) at the roofline knee."""
        return (self.tc_tflops * 1e12) / (self.hbm_bw_gbs * 1e9)


A800_80GB = HardwareSpec(
    name="A800-80GB",
    tc_tflops=312.0,        # bf16 dense
    hbm_bw_gbs=1555.0,      # 1.555 TB/s, A800 = A100 with NVLink-throttle
    pcie_bw_gbs=25.0,       # PCIe Gen4 x16 sustained
    sm_count=108,
    mem_gb=80.0,
)

A100_80GB = HardwareSpec(
    name="A100-80GB",
    tc_tflops=312.0,
    hbm_bw_gbs=2039.0,      # 2.04 TB/s
    pcie_bw_gbs=25.0,
    sm_count=108,
    mem_gb=80.0,
)

H100_80GB = HardwareSpec(
    name="H100-80GB-SXM",
    tc_tflops=989.0,        # bf16 dense, no sparsity
    hbm_bw_gbs=3350.0,      # 3.35 TB/s HBM3
    pcie_bw_gbs=63.0,       # PCIe Gen5 x16 sustained
    sm_count=132,
    mem_gb=80.0,
)


# ---------------------------------------------------------------------------
# Model spec for SynapForge HybridBlock.
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Shape parameters for HybridBlock + LM head.

    Fields
    ------
    d           : hidden dim (default 1280, Run 7).
    n_layers    : number of HybridBlocks.
    loop_depth  : LoopLM repeat count per block (Run 7 = 2).
    ffn_ratio   : SwiGLU hidden / d (Run 7 = 3.0).
    seq_len     : tokens per micro-batch.
    batch_size  : micro-batch size.
    grad_accum  : gradient accumulation factor.
    vocab       : LM-head output dim (Run 7 = 151936, Qwen2.5 tokenizer).
    dtype_bytes : bytes per activation/weight (bf16 = 2).
    n_params    : total parameter count (auto-computed if 0).
    """

    d: int = 1280
    n_layers: int = 16
    loop_depth: int = 2
    ffn_ratio: float = 3.0
    seq_len: int = 1024
    batch_size: int = 32
    grad_accum: int = 2
    vocab: int = 151936
    dtype_bytes: int = 2
    n_params: int = 0

    def __post_init__(self) -> None:
        if self.n_params == 0:
            self.n_params = self._compute_n_params()

    @property
    def ffn_hidden(self) -> int:
        return int(self.d * self.ffn_ratio)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len

    def _compute_n_params(self) -> int:
        """Approximate parameter count (embed + n_layers*HybridBlock + lm_head).

        HybridBlock per layer (no biases counted separately, dwarfed by GEMMs):
          rmsnorm (x2)      : 2 * d
          cfc_step          : 2 * d * d  (delta_proj + b_proj)
          plif              : ~3 * d  (tau_log + threshold + alpha; tiny)
          swiglu            : 3 * d * ffn_hidden
          sew_shortcut      : d * d  (optional, run7 has it)
          ln_residual fudge : 0
        """
        d = self.d
        h = self.ffn_hidden
        per_layer = 4 * d * d + 3 * d * h + 2 * d + 3 * d
        embed = self.vocab * d
        lm_head = d * self.vocab
        return int(embed + lm_head + self.n_layers * per_layer * self.loop_depth)


# ---------------------------------------------------------------------------
# Per-stage FLOPs / bytes accounting.
# ---------------------------------------------------------------------------


@dataclass
class StageRoofline:
    """One row of the roofline table -- one logical stage of one step.

    All quantities are *per-step* (not per-token) so they can be summed.
    """

    name: str
    flops: float
    bytes_io: float
    gpu_resident_only: bool       # if False, bytes cross PCIe each step
    gemm_dominated: bool          # if True, can theoretically hit TC peak

    # Filled in by ``compute_roofline`` after a HardwareSpec is bound:
    ai: float = 0.0
    ms_compute: float = 0.0
    ms_hbm: float = 0.0
    ms_pcie: float = 0.0
    binding: str = ""             # "compute" | "hbm" | "pcie"
    # Wall-clock time for this stage given the roofs. All three
    # resources (TC, HBM, PCIe) must each complete their share -- so
    # wall = max(ms_compute, ms_hbm, ms_pcie). Kept under the historical
    # name ``ms_min`` because it's the *minimum achievable* wall-time.
    ms_min: float = 0.0


def _hybridblock_fwd_flops(model: ModelSpec) -> float:
    """FLOPs per token for one HybridBlock fwd (Run 7 architecture).

    Components (matmuls only count; rmsnorm/plif/cfc-step are O(d) and
    dwarfed):
        cfc_step (delta_proj + b_proj)  : 2 * 2 * d * d = 4 * d^2
        swiglu (gate + up + down)       : 6 * d * h
        sew_shortcut (residual matmul)  : 2 * d * d
    """
    d = model.d
    h = model.ffn_hidden
    return 4.0 * d * d + 6.0 * d * h + 2.0 * d * d


def _hybridblock_fwd_bytes_per_step(model: ModelSpec, tokens_per_step: int) -> float:
    """Bytes-moved *per step* for ONE HybridBlock fwd (caller multiplies
    by n_blocks).

    Each GEMM moves (W + X + Y) bytes. For batched GEMM with B*T tokens
    sharing the same weight matrix:
        bytes = W (read once per step) + X*tokens + Y*tokens

    Per block:
        cfc_step (delta + b)   : 2*(d*d) weight + 2*(2*d)*tokens act
        swiglu (gate + up)     : 2*(d*h) weight + 2*(d+h)*tokens act
        swiglu (down)          : 1*(h*d) weight + 1*(h+d)*tokens act
        sew_shortcut           : 1*(d*d) weight + 1*(2*d)*tokens act

    Activation reuse across the loop_depth iterations is left to the
    caller; this is one-block-one-pass.
    """
    d = model.d
    h = model.ffn_hidden
    db = float(model.dtype_bytes)
    # Weights read once per fwd per block (PCIe-side cached in HBM):
    # cfc 2*d*d + swiglu (gate+up) 2*d*h + swiglu down h*d + sew d*d
    weight_bytes = (3 * d * d + 3 * d * h) * db
    # Activations move per token: every sub-op reads d-vector input and
    # writes d-or-h-vector output. Approximate as 8*d traffic per token
    # (intermediate ffn h is bigger but most reuses come right back to d).
    act_bytes_per_tok = 8 * d * db + 2 * h * db  # ~ d + h round trip
    return weight_bytes + act_bytes_per_tok * float(tokens_per_step)


def _lm_head_fwd_flops(model: ModelSpec) -> float:
    return 2.0 * model.d * model.vocab


def _lm_head_fwd_bytes(model: ModelSpec) -> float:
    """LM head moves W (d*V) once per step; X+Y are small per token.
    Per-token bytes: V (output logits) + d (input). W cost amortizes
    over batch but we report per-token.
    """
    d = model.d
    V = model.vocab
    db = float(model.dtype_bytes)
    return (d + V) * db + (d * V) * db / max(1, model.batch_size * model.seq_len)


def _embed_fwd_flops(model: ModelSpec) -> float:
    # gather, no GEMM. Treat as 0 FLOPs.
    return 0.0


def _embed_fwd_bytes(model: ModelSpec) -> float:
    # one row of d-bytes per token from the embed matrix (V*d)
    return float(model.d) * model.dtype_bytes


def _data_loader_bytes_per_token(model: ModelSpec) -> float:
    """Bytes the data loader pushes from CPU/disk to GPU per token.

    Tokens are int32 (4B) per ID + int32 mask (4B) = 8 B/tok over PCIe.
    Plus the KD-distill teacher logits: --kd-topk 2048 in Run 7, 2048 *
    2 (bf16) + 2048 * 4 (int idx) = 12_288 bytes/tok every kd_every=4
    steps -> 3072 bytes/tok amortized. Big.
    """
    base = 8.0
    kd_topk = 2048
    kd_every = 4
    kd_bytes = kd_topk * (model.dtype_bytes + 4) / kd_every  # bf16 prob + int4 idx
    return base + kd_bytes


def _optimizer_bytes_per_step(model: ModelSpec, cpu_offload: bool = True) -> float:
    """AdamW moves (param, m, v, grad) every step. With cpu_offload-optim
    they cross PCIe twice (D2H grad, H2D updated param) = 2 * (4*P*4B
    fp32 master) bytes. Without offload, weights stay GPU-resident.
    """
    P = model.n_params
    if cpu_offload:
        # grad bf16: 2B, master fp32 weight: 4B; both cross PCIe each step.
        # Adam state (m, v) is fp32 on CPU, doesn't cross.
        return 2.0 * P * (model.dtype_bytes + 4)
    return 0.0  # in-place GPU update


# ---------------------------------------------------------------------------
# Public roofline computation.
# ---------------------------------------------------------------------------


@dataclass
class RooflineResult:
    """Full roofline output for a (model, hw, options) triple."""

    model: ModelSpec
    hw: HardwareSpec
    cpu_offload_optim: bool
    grad_ckpt: bool

    stages: List[StageRoofline] = field(default_factory=list)
    ms_step_total: float = 0.0     # sum over stages of min(compute, hbm, pcie)
    ms_step_compute: float = 0.0   # if everything were TC-bound
    ms_step_hbm: float = 0.0       # if everything were HBM-bound
    ms_step_pcie: float = 0.0      # if everything were PCIe-bound
    tok_per_sec_compute: float = 0.0
    tok_per_sec_hbm: float = 0.0
    tok_per_sec_pcie: float = 0.0
    tok_per_sec_total: float = 0.0    # the true upper bound = min of all
    binding_stage: str = ""           # which stage's binding regime sets the step
    binding_regime: str = ""          # "compute" | "hbm" | "pcie"

    def to_dict(self) -> Dict:
        return {
            "model": asdict(self.model),
            "hw": asdict(self.hw),
            "cpu_offload_optim": self.cpu_offload_optim,
            "grad_ckpt": self.grad_ckpt,
            "stages": [asdict(s) for s in self.stages],
            "ms_step_total": self.ms_step_total,
            "ms_step_compute": self.ms_step_compute,
            "ms_step_hbm": self.ms_step_hbm,
            "ms_step_pcie": self.ms_step_pcie,
            "tok_per_sec_compute": self.tok_per_sec_compute,
            "tok_per_sec_hbm": self.tok_per_sec_hbm,
            "tok_per_sec_pcie": self.tok_per_sec_pcie,
            "tok_per_sec_total": self.tok_per_sec_total,
            "binding_stage": self.binding_stage,
            "binding_regime": self.binding_regime,
        }


def _resolve_stage(stage: StageRoofline, hw: HardwareSpec) -> None:
    """Fill in ai, ms_compute, ms_hbm, ms_pcie, binding for a stage."""
    flops = stage.flops
    bytes_io = stage.bytes_io
    if bytes_io <= 0:
        bytes_io = 1.0  # avoid div0 for FLOP-only stages
    stage.ai = flops / bytes_io

    # Compute roof: time at peak TC TFLOPS. For non-GEMM stages this is
    # tiny (~ 0 ms) and won't bind; we still compute it so the table has
    # a real number not `inf`. For zero-FLOP stages (gather, copy) it's
    # exactly 0.
    if flops > 0:
        stage.ms_compute = flops / (hw.tc_tflops * 1e12) * 1000.0
    else:
        stage.ms_compute = 0.0

    # HBM roof: bytes that can stay GPU-resident.
    stage.ms_hbm = bytes_io / (hw.hbm_bw_gbs * 1e9) * 1000.0

    # PCIe roof: only stages that cross the host link.
    if not stage.gpu_resident_only:
        stage.ms_pcie = bytes_io / (hw.pcie_bw_gbs * 1e9) * 1000.0
    else:
        stage.ms_pcie = 0.0  # PCIe doesn't gate this stage

    # Within a single stage, compute and HBM and PCIe traffic must each
    # complete. Wall-time = max(t_compute, t_hbm, t_pcie). The "binding"
    # resource is the one with the largest of the three.
    candidates: List[Tuple[float, str]] = [
        (stage.ms_compute, "compute"),
        (stage.ms_hbm, "hbm"),
    ]
    if not stage.gpu_resident_only:
        candidates.append((stage.ms_pcie, "pcie"))
    ms_max, binding = max(candidates, key=lambda x: x[0])
    stage.ms_min = ms_max  # name kept for back-compat; actually max-of-three
    stage.binding = binding


def compute_roofline(
    model: ModelSpec,
    hw: HardwareSpec,
    cpu_offload_optim: bool = True,
    grad_ckpt: bool = True,
    bwd_to_fwd_ratio: float = 2.0,
) -> RooflineResult:
    """Build the full roofline for one (model, hw) pair.

    Parameters
    ----------
    model              : ``ModelSpec`` instance.
    hw                 : ``HardwareSpec`` instance.
    cpu_offload_optim  : whether AdamW state lives on CPU (Run 7 = True).
    grad_ckpt          : whether activations are recomputed in bwd
                         (Run 7 = True). When True, fwd is run twice;
                         when False, only once. We do NOT add a second
                         fwd here -- we just adjust HBM bytes.
    bwd_to_fwd_ratio   : 2.0 by convention (linear bwd is roughly 2x
                         fwd FLOPs because grad_x and grad_W are each
                         ~ fwd cost).

    Returns
    -------
    RooflineResult with per-stage timings and aggregated tok/s.
    """
    if not isinstance(model, ModelSpec):
        raise TypeError(f"model must be ModelSpec, got {type(model)}")
    if not isinstance(hw, HardwareSpec):
        raise TypeError(f"hw must be HardwareSpec, got {type(hw)}")

    res = RooflineResult(
        model=model, hw=hw,
        cpu_offload_optim=cpu_offload_optim, grad_ckpt=grad_ckpt,
    )

    tokens = float(model.tokens_per_step)
    n_blocks = model.n_layers * model.loop_depth

    # ----- Stage 1: data loader (CPU disk read + decode + push to GPU)
    bytes_data = _data_loader_bytes_per_token(model) * tokens
    res.stages.append(StageRoofline(
        name="data_loader",
        flops=0.0,                      # no GEMM
        bytes_io=bytes_data,
        gpu_resident_only=False,        # crosses PCIe
        gemm_dominated=False,
    ))

    # ----- Stage 2: embedding lookup (gather, GPU-resident)
    bytes_embed = _embed_fwd_bytes(model) * tokens
    res.stages.append(StageRoofline(
        name="embed_fwd",
        flops=0.0,                      # gather, no GEMM
        bytes_io=bytes_embed,
        gpu_resident_only=True,
        gemm_dominated=False,
    ))

    # ----- Stage 3: HybridBlock fwd (n_layers * loop_depth blocks)
    fwd_per_block_flops = _hybridblock_fwd_flops(model) * tokens
    fwd_per_block_bytes = _hybridblock_fwd_bytes_per_step(
        model, tokens_per_step=int(tokens)
    )
    # Aggregate across all blocks
    fwd_total_flops = fwd_per_block_flops * n_blocks
    fwd_total_bytes = fwd_per_block_bytes * n_blocks
    if grad_ckpt:
        # Activation recomputation: bwd repeats fwd -> 1.5x fwd HBM
        # (we still count fwd_total_bytes here, the second pass is in bwd)
        pass
    res.stages.append(StageRoofline(
        name="hybridblock_fwd",
        flops=fwd_total_flops,
        bytes_io=fwd_total_bytes,
        gpu_resident_only=True,
        gemm_dominated=True,
    ))

    # ----- Stage 4: LM head fwd (huge V=151936 -> compute heavy)
    lm_fwd_flops = _lm_head_fwd_flops(model) * tokens
    lm_fwd_bytes = _lm_head_fwd_bytes(model) * tokens
    res.stages.append(StageRoofline(
        name="lm_head_fwd",
        flops=lm_fwd_flops,
        bytes_io=lm_fwd_bytes,
        gpu_resident_only=True,
        gemm_dominated=True,
    ))

    # ----- Stage 5: HybridBlock bwd (~2x fwd FLOPs, ~2x fwd bytes if no ckpt)
    bwd_total_flops = fwd_total_flops * bwd_to_fwd_ratio
    bwd_total_bytes = fwd_total_bytes * (1.5 if grad_ckpt else 2.0)
    res.stages.append(StageRoofline(
        name="hybridblock_bwd",
        flops=bwd_total_flops,
        bytes_io=bwd_total_bytes,
        gpu_resident_only=True,
        gemm_dominated=True,
    ))

    # ----- Stage 6: LM head bwd
    lm_bwd_flops = lm_fwd_flops * bwd_to_fwd_ratio
    lm_bwd_bytes = lm_fwd_bytes * 2.0
    res.stages.append(StageRoofline(
        name="lm_head_bwd",
        flops=lm_bwd_flops,
        bytes_io=lm_bwd_bytes,
        gpu_resident_only=True,
        gemm_dominated=True,
    ))

    # ----- Stage 7: optimizer (AdamW, possibly CPU-offload)
    opt_bytes = _optimizer_bytes_per_step(model, cpu_offload=cpu_offload_optim)
    # AdamW math is FLOPs ~ 8 * P (mul/add per moment). Dwarfed by GEMM.
    opt_flops = 8.0 * model.n_params
    res.stages.append(StageRoofline(
        name="optimizer_step",
        flops=opt_flops,
        bytes_io=opt_bytes,
        gpu_resident_only=not cpu_offload_optim,
        gemm_dominated=False,
    ))

    # Bind hardware to each stage
    for s in res.stages:
        _resolve_stage(s, hw)

    # Aggregate -- if everything were compute-bound:
    res.ms_step_compute = sum(
        s.flops / (hw.tc_tflops * 1e12) * 1000.0
        for s in res.stages if s.gemm_dominated
    )
    res.ms_step_hbm = sum(
        s.bytes_io / (hw.hbm_bw_gbs * 1e9) * 1000.0
        for s in res.stages
    )
    res.ms_step_pcie = sum(
        s.bytes_io / (hw.pcie_bw_gbs * 1e9) * 1000.0
        for s in res.stages if not s.gpu_resident_only
    )

    # Total step is the sum of each stage's wall-time min(compute, hbm,
    # pcie). This is the *true* upper bound -- you can't skip a stage.
    res.ms_step_total = sum(s.ms_min for s in res.stages)
    res.tok_per_sec_total = tokens / (res.ms_step_total / 1000.0) if res.ms_step_total > 0 else 0.0
    res.tok_per_sec_compute = tokens / (res.ms_step_compute / 1000.0) if res.ms_step_compute > 0 else 0.0
    res.tok_per_sec_hbm = tokens / (res.ms_step_hbm / 1000.0) if res.ms_step_hbm > 0 else 0.0
    res.tok_per_sec_pcie = tokens / (res.ms_step_pcie / 1000.0) if res.ms_step_pcie > 0 else 0.0

    # Binding stage = the one whose ms_min is largest fraction of total
    if res.stages:
        slow = max(res.stages, key=lambda s: s.ms_min)
        res.binding_stage = slow.name
        res.binding_regime = slow.binding

    return res


# ---------------------------------------------------------------------------
# Pretty-print helpers (no rich/pandas dep).
# ---------------------------------------------------------------------------


def format_roofline_table(res: RooflineResult) -> str:
    """Return a plain-text breakdown matching docs/SATURATION_REPORT.md."""
    lines: List[str] = []
    lines.append(
        f"# Roofline -- {res.hw.name} | "
        f"d={res.model.d} L={res.model.n_layers}x{res.model.loop_depth} "
        f"V={res.model.vocab} bs={res.model.batch_size}*{res.model.grad_accum} "
        f"T={res.model.seq_len}"
    )
    lines.append(
        f"# tokens/step = {res.model.tokens_per_step}; "
        f"params = {res.model.n_params/1e6:.1f}M"
    )
    lines.append(
        f"# HW: TC={res.hw.tc_tflops:.0f}TF, HBM={res.hw.hbm_bw_gbs:.0f}GB/s, "
        f"PCIe={res.hw.pcie_bw_gbs:.0f}GB/s, ridge AI={res.hw.ridge_ai:.1f}"
    )
    lines.append("")
    hdr = (
        f"{'stage':<22} {'GFLOPs':>10} {'MB':>10} {'AI':>8} "
        f"{'ms_C':>8} {'ms_HBM':>8} {'ms_PCI':>8} {'ms_min':>8} {'bind':<8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in res.stages:
        lines.append(
            f"{s.name:<22} {s.flops/1e9:>10.2f} {s.bytes_io/1e6:>10.2f} "
            f"{s.ai:>8.1f} {s.ms_compute:>8.2f} {s.ms_hbm:>8.2f} "
            f"{s.ms_pcie:>8.2f} {s.ms_min:>8.2f} {s.binding:<8}"
        )
    lines.append("-" * len(hdr))
    lines.append(
        f"step total: {res.ms_step_total:.2f} ms => "
        f"{res.tok_per_sec_total:.0f} tok/s | "
        f"binding: {res.binding_stage} ({res.binding_regime})"
    )
    lines.append(
        f"  if all compute-bound:  {res.ms_step_compute:.2f} ms / "
        f"{res.tok_per_sec_compute:.0f} tok/s"
    )
    lines.append(
        f"  if all HBM-bound:      {res.ms_step_hbm:.2f} ms / "
        f"{res.tok_per_sec_hbm:.0f} tok/s"
    )
    lines.append(
        f"  if all PCIe-bound:     {res.ms_step_pcie:.2f} ms / "
        f"{res.tok_per_sec_pcie:.0f} tok/s"
    )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover -- CLI smoke
    """Print A800 roofline for Run 7's exact config."""
    model = ModelSpec(
        d=1280, n_layers=16, loop_depth=2, ffn_ratio=3.0,
        seq_len=1024, batch_size=32, grad_accum=2, vocab=151936,
    )
    res = compute_roofline(model, A800_80GB, cpu_offload_optim=True, grad_ckpt=True)
    print(format_roofline_table(res))


if __name__ == "__main__":
    main()
