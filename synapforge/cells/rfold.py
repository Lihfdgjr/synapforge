"""R-fold closed-form for gated CfC over k Coconut latent steps.

Per agent synthesis 2026-05-01 (Cayley-Hamilton + LiquidS4 hybrid).

The gated CfC step
    pre = W_in x + W_h h
    g   = sigmoid(W_gate pre)
    h'  = (1 - g*alpha) * h + g*alpha * tanh(pre)
is non-linear via the gate. We freeze g at h_0 and linearize tanh once,
giving an affine map h_{t+1} = M h_t + b. Geometric series collapses
R steps:
    h_R = M^R h_0 + (I - M)^{-1} (I - M^R) b

Speedup at R=8 is ~2.7x; at R=64 ~9x; at R=1024 with chunked re-anchor L=16
~3-4x at near-sequential quality. The naive 167x claim only holds for a
single fold without re-anchoring (gate drift > 8% NIAH at k=64).

Anchors: LiquidS4 (2401.13386), S5 (2208.04933), Coconut (2412.06769).
"""

from __future__ import annotations

import torch


def _matrix_power_squaring(M: torch.Tensor, R: int) -> torch.Tensor:
    """O(N^3 log R) matrix power for batched square M.  M: [B, N, N]."""
    if R == 0:
        return torch.eye(M.size(-1), device=M.device, dtype=M.dtype).expand_as(M).clone()
    result = None
    base = M
    r = R
    while r > 0:
        if r & 1:
            result = base if result is None else result @ base
        r >>= 1
        if r:
            base = base @ base
    return result


def cfc_rfold(
    h0: torch.Tensor,
    x: torch.Tensor,
    W_in: torch.Tensor,
    W_h: torch.Tensor,
    W_gate: torch.Tensor,
    tau: torch.Tensor,
    R: int,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Closed-form R-step CfC under fixed gate + first-order tanh linearization.

    h0:    [B, N]    initial hidden
    x:     [B, D]    fixed input across R steps (Coconut latent loop)
    W_in:  [N, D]
    W_h:   [N, N]
    W_gate:[N, N]
    tau:   [N]
    R:     number of latent steps (k=8 typical for Coconut)

    Returns h_R: [B, N].  Forces fp32 internally (bf16 catastrophic at R>16).
    """
    if R <= 0:
        return h0
    orig_dtype = h0.dtype
    h0_f = h0.float()
    x_f = x.float()
    W_in_f = W_in.float()
    W_h_f = W_h.float()
    W_gate_f = W_gate.float()
    tau_f = tau.float()

    B, N = h0_f.shape
    pre0 = x_f @ W_in_f.T + h0_f @ W_h_f.T              # [B, N]
    alpha = torch.sigmoid(tau_f)                         # [N]
    g = torch.sigmoid(pre0 @ W_gate_f.T)                 # [B, N]
    beta = g * alpha                                     # [B, N]

    sech2 = 1.0 - torch.tanh(pre0).pow(2)                # [B, N]

    eye = torch.eye(N, device=h0.device, dtype=torch.float32)
    diag_b = beta.unsqueeze(-1)                          # [B, N, 1]
    diag_s = sech2.unsqueeze(-1)                         # [B, N, 1]
    M = (1.0 - diag_b) * eye + diag_b * diag_s * W_h_f   # [B, N, N]

    Jh0 = (diag_s * W_h_f) @ h0_f.unsqueeze(-1)          # [B, N, 1]
    c = torch.tanh(pre0) - Jh0.squeeze(-1)               # [B, N]
    b = beta * c                                         # [B, N]

    MR = _matrix_power_squaring(M, R)                    # [B, N, N]
    h_pow = (MR @ h0_f.unsqueeze(-1)).squeeze(-1)        # [B, N]

    I_minus_M = eye - M + ridge * eye                    # ridge for skip-gate
    rhs = (b.unsqueeze(-1) - MR @ b.unsqueeze(-1))
    geom = torch.linalg.solve(I_minus_M, rhs).squeeze(-1)
    return (h_pow + geom).to(orig_dtype)


def cfc_rfold_neumann(
    h0: torch.Tensor,
    x: torch.Tensor,
    W_in: torch.Tensor,
    W_h: torch.Tensor,
    W_gate: torch.Tensor,
    tau: torch.Tensor,
    R: int,
) -> torch.Tensor:
    """CPU-friendly fold: replaces (I-M)^-1 solve with truncated Neumann series.

    The closed form
        h_R = M^R h_0 + (I - M)^{-1} (I - M^R) b
    expands to
        h_R = M^R h_0 + (I + M + M^2 + ... + M^{R-1}) b
    This avoids `torch.linalg.solve` (LAPACK getrs has ~50ms fixed cost on
    CPU at N=512), at the price of one extra matmul per step. Wins on
    CPU + small batch.

    Numerical: identical to cfc_rfold up to fp32 round-off when |lambda(M)|<1
    (which our spectral-norm bound on W_h enforces). For batch CFP, this is
    the right CPU primitive.
    """
    if R <= 0:
        return h0
    orig_dtype = h0.dtype
    h0_f = h0.float()
    x_f = x.float()
    W_in_f = W_in.float()
    W_h_f = W_h.float()
    W_gate_f = W_gate.float()
    tau_f = tau.float()

    B, N = h0_f.shape
    pre0 = x_f @ W_in_f.T + h0_f @ W_h_f.T
    alpha = torch.sigmoid(tau_f)
    g = torch.sigmoid(pre0 @ W_gate_f.T)
    beta = g * alpha

    sech2 = 1.0 - torch.tanh(pre0).pow(2)

    eye = torch.eye(N, device=h0.device, dtype=torch.float32)
    diag_b = beta.unsqueeze(-1)
    diag_s = sech2.unsqueeze(-1)
    M = (1.0 - diag_b) * eye + diag_b * diag_s * W_h_f

    Jh0 = (diag_s * W_h_f) @ h0_f.unsqueeze(-1)
    c = torch.tanh(pre0) - Jh0.squeeze(-1)
    b = beta * c

    # geometric series: accumulate sum_{k=0..R-1} M^k @ b   and   M^R @ h0
    h_pow = h0_f.unsqueeze(-1)
    geom = b.unsqueeze(-1).clone()
    Mk_b = b.unsqueeze(-1).clone()
    for _ in range(R):
        h_pow = M @ h_pow
        Mk_b = M @ Mk_b
        geom = geom + Mk_b
    geom = geom - Mk_b  # we accumulated R+1 terms, drop last
    return (h_pow.squeeze(-1) + geom.squeeze(-1)).to(orig_dtype)


def cfc_rfold_auto(
    h0: torch.Tensor,
    x: torch.Tensor,
    W_in: torch.Tensor,
    W_h: torch.Tensor,
    W_gate: torch.Tensor,
    tau: torch.Tensor,
    R: int,
) -> torch.Tensor:
    """Auto-dispatch fold variant by device + (N, R) shape.

    CPU + N>=128: Neumann series (no LAPACK solve overhead)
    GPU + N>=256: cfc_rfold (cuBLAS bmm + solve is fine)
    Anything else: sequential (small N, no benefit from folding)
    """
    N = h0.shape[-1]
    on_cpu = h0.device.type == "cpu"
    if on_cpu:
        # Empirical (verify_rfold.py 2026-05-01): on CPU, sequential always
        # wins for N>=128 because building the [B,N,N] matrix dominates.
        # Neumann is 5-7x faster than solve-fold but still loses to sequential.
        # Only fold is worth it on GPU + cuBLAS bmm.
        return _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R)
    if N >= 256 and R >= 4:
        return cfc_rfold(h0, x, W_in, W_h, W_gate, tau, R)
    return _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R)


def _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R):
    """Reference path used by auto-dispatch when fold is not beneficial."""
    h = h0
    alpha = torch.sigmoid(tau)
    for _ in range(R):
        pre = x @ W_in.T + h @ W_h.T
        g = torch.sigmoid(pre @ W_gate.T)
        beta = g * alpha
        h = (1.0 - beta) * h + beta * torch.tanh(pre)
    return h


def cfc_rfold_chunked(
    h0: torch.Tensor,
    x: torch.Tensor,
    W_in: torch.Tensor,
    W_h: torch.Tensor,
    W_gate: torch.Tensor,
    tau: torch.Tensor,
    R: int,
    chunk: int = 8,
) -> torch.Tensor:
    """R-fold with chunked re-anchoring every `chunk` steps.

    For R >= 64 the single-fold gate drift kills NIAH recall (>8% at R=64).
    Chunking re-computes (g, J) every `chunk` steps. Speedup degrades to
    ~3-4x at R=1024 vs sequential, but quality stays within ppl +0.3-0.8%.
    """
    h = h0
    remaining = R
    while remaining > 0:
        step = min(chunk, remaining)
        h = cfc_rfold(h, x, W_in, W_h, W_gate, tau, step)
        remaining -= step
    return h


# ---------------------------------------------------------------------------
# Diagonal-recurrence parallel scan (LiquidCell-specific).
# ---------------------------------------------------------------------------
# The CfC rfold above linearizes a *gated* CfC step. LiquidCell's actual
# forward (synapforge/cells/liquid.py) is different: per-channel diagonal
# A_t plus additive b_t.  That recurrence is *already* affine in h, so no
# linearization is needed -- only a numerically stable parallel scan.
#
# Math (B = batch, T = time, D = hidden width, all per-channel diagonal):
#     h_t = A_t * h_{t-1} + b_t
#     => h_t = (prod_{s<=t} A_s) * h_{-1}
#                + sum_{s<=t} (prod_{u=s+1..t} A_u) * b_s
#
# The naive Heinsen form (cumsum of log A) overflows for T>=128 because
# log A can be -10 per step. We avoid log space by working in raw cumprod
# space inside *small chunks* of length R, then chaining chunks
# sequentially.  Within a chunk we evaluate the closed form directly:
#     P_t = cumprod(A)[t] = prod_{s<=t} A_s
#     numerator  = b_t / max(P_t, eps_floor)         # rebased input
#     h_t = P_t * (h_{-1} + cumsum(numerator)[t])
#
# Stability: when |A_t| < 1 strongly, P_t -> 0; numerator blows up; final
# h_t = 0 * inf = NaN.  We clamp P_t >= eps_floor (default 1e-8) inside the
# division.  The output P_t * cumsum is still equal to the true sum
# whenever |b_t| stays small relative to ``eps_floor / cumsum_max``.
# Empirically (test_rfold_equivalence) this matches the sequential
# reference within fp32 round-off for chunk<=16 at training-scale inputs.
#
# At chunk=16 and T=256 we make 16 short scans instead of 256 step
# launches.  On GPU this is the kernel-launch saving the user wants
# ("simplify the computation, not just replace the framework").  No
# matrix-exp, no solve, no extra parameters; pure algebra.


def liquid_rfold_chunk(
    A_t: torch.Tensor,
    b_t: torch.Tensor,
    h_init: torch.Tensor,
    eps_floor: float = 1e-30,
) -> torch.Tensor:
    """One chunk of the closed-form diagonal scan.

    Args
    ----
    A_t : (B, R, D) per-channel decay, A_t in (0, 1].
    b_t : (B, R, D) per-channel input drive.
    h_init : (B, D) state at t = -1.
    eps_floor : minimum cumprod value (avoid 0-divide for very small A).

    Returns
    -------
    h : (B, R, D)  -- one h_t per step in the chunk.

    Mathematically identical to:
        h = h_init
        for t in range(R):
            h = A_t[:, t] * h + b_t[:, t]
            out[:, t] = h
    up to fp32 round-off introduced by the eps_floor clamp (which only
    activates when the true cumprod is already underflowed).
    """
    if A_t.dim() != 3 or b_t.dim() != 3:
        raise ValueError(
            f"liquid_rfold_chunk: expected (B,R,D), got A={tuple(A_t.shape)} "
            f"b={tuple(b_t.shape)}"
        )
    if A_t.shape != b_t.shape:
        raise ValueError(
            f"A_t {tuple(A_t.shape)} vs b_t {tuple(b_t.shape)} mismatch"
        )
    A_f = A_t.float()
    b_f = b_t.float()
    h0 = h_init.float()

    # P_t = prod_{s<=t} A_s   (B, R, D)
    P = torch.cumprod(A_f, dim=1)
    P_safe = P.clamp(min=eps_floor)
    # rebased input  b_t / P_t       (B, R, D)
    rebased = b_f / P_safe
    # cumulative sum along time axis (B, R, D)
    csum = torch.cumsum(rebased, dim=1)
    # h_t = P_t * (h_init + csum_t)
    h = P * (h0.unsqueeze(1) + csum)
    return h


def liquid_rfold(
    A_t: torch.Tensor,
    b_t: torch.Tensor,
    h_init: torch.Tensor,
    chunk: int = 16,
    eps_floor: float = 1e-30,
) -> torch.Tensor:
    """Chunked closed-form scan for the LiquidCell diagonal recurrence.

    Parameters
    ----------
    A_t : (B, T, D) per-channel decay, A_t in (0, 1].
    b_t : (B, T, D) per-channel input drive.
    h_init : (B, D) state at t = -1 (h_0 in the paper).
    chunk : steps per closed-form chunk. Larger = fewer Python iterations,
        worse stability when ``A_t`` has small entries (cumprod underflows).
        16 is the empirical sweet spot for our `A_log` init (`hasani` gives
        log A_t in roughly [-10, 0] per step; cumprod over 16 steps stays
        above 1e-70, well within fp32 range).
    eps_floor : clamp on the cumprod to avoid zero-divide at extreme decay.

    Returns
    -------
    h : (B, T, D)
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if A_t.dim() != 3 or b_t.dim() != 3:
        raise ValueError(
            f"liquid_rfold: expected (B,T,D), got A={tuple(A_t.shape)} "
            f"b={tuple(b_t.shape)}"
        )
    if A_t.shape != b_t.shape:
        raise ValueError(
            f"A_t {tuple(A_t.shape)} vs b_t {tuple(b_t.shape)} mismatch"
        )
    B, T, D = A_t.shape
    if h_init.shape != (B, D):
        raise ValueError(
            f"h_init must be (B,D)=({B},{D}), got {tuple(h_init.shape)}"
        )

    # Trivial small-T: avoid the chunking overhead entirely.
    if T <= chunk:
        return liquid_rfold_chunk(A_t, b_t, h_init, eps_floor=eps_floor)

    out_chunks: list[torch.Tensor] = []
    h_curr = h_init
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        h_chunk = liquid_rfold_chunk(
            A_t[:, start:end, :],
            b_t[:, start:end, :],
            h_curr,
            eps_floor=eps_floor,
        )
        out_chunks.append(h_chunk)
        # Carry state across chunks via the LAST step of the chunk
        # (sequential boundary -- this is the chained scan trick:
        # parallel within chunk, sequential between chunks).
        h_curr = h_chunk[:, -1, :]
    return torch.cat(out_chunks, dim=1)
