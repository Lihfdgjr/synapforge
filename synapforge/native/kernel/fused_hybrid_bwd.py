"""Fused backward Triton kernel for the entire ``HybridBlock`` chain.

Like ``fused_hybrid_fwd.py``, this file has ZERO ``import torch``. It
ships closed-form gradient kernels that consume only the saved-for-bwd
intermediates (``h_pre, h_post, v_pre, v_post, spike, rstd1, rstd2,
gate_act, ffn_act, ...``) and produce dL/d(input) plus dL/d(weights).

Closed-form derivation (forward repeated for reference)
=======================================================

Forward chain (one HybridBlock invocation, ignoring residuals
applied in glue):

    y1[b,t,d]    = x[b,t,d] * rstd1[b,t] * w_norm1[d]
    delta[b,t,d] = softplus(W_d y1 + b_d)            # GEMM in glue
    bvec[b,t,d]  = delta * (W_b y1 + b_b)            # GEMM in glue
    A_t          = exp(-delta * exp(A_log))
    h_pre[b,t,d] = A_t * h_pre_post[b,t-1,d] + bvec   # scan
    h_post[b,t,d]= tanh(h_pre)
    v_pre[b,t,d] = decay * v_post[b,t-1,d] + (1-decay) * h_post
    s[b,t,d]     = Heaviside(v_pre - thr)
    v_post[b,t,d]= v_pre - s * thr
    spike_in     = s + h_post                          # SEW
    syn          = SparseSynapse(spike_in)             # GEMM in glue
    gate_pre     = W_g spike_in + b_g                  # GEMM in glue
    gate         = sigmoid(gate_pre)
    gated        = syn * gate
    x1           = x + gated
    y2           = x1 * rstd2 * w_norm2                # second RMSNorm
    g_pre        = W_gate_ffn @ y2                     # GEMM in glue
    up_pre       = W_up @ y2                           # GEMM in glue
    silu_g       = g_pre * sigmoid(g_pre)
    ffn_act      = silu_g * up_pre
    ffn_out      = W_down @ ffn_act                    # GEMM in glue
    x_final      = x1 + ffn_out

Backward (downstream feeds dL/dx_final):

(stage F2: residual #2)
    dL/dx1     += dL/dx_final
    dL/dffn_out = dL/dx_final
    dL/d(W_down) = dL/dffn_out @ ffn_act.T            # GEMM in glue
    dL/d(ffn_act) = W_down.T @ dL/dffn_out             # GEMM in glue

(stage F1: SwiGLU closed-form)
    For y = silu(g) * up:
        dy/dg = sigmoid(g) * (1 + g * (1 - sigmoid(g))) * up
              = silu_g * up * (1 + g - silu_g)/g
              -- actually we use the DIRECT form:
              dy/dg = up * (sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g)))
              dy/dup = silu_g
    So:
        dL/dg_pre  = dL/dffn_act * up_pre * sig_g * (1 + g_pre * (1 - sig_g))
        dL/dup_pre = dL/dffn_act * silu_g
    Then GEMM through W_gate_ffn / W_up (in glue) to get dL/dy2 and
    dL/d(W_gate_ffn), dL/d(W_up).

(stage RN2: RMSNorm bwd, closed-form)
    For y = x * rstd * w with rstd = 1/sqrt(mean(x^2) + eps):
        dL/dw = sum_{b,t} dL/dy * x * rstd
        dL/dx = (rstd * w) * dL/dy
              - (rstd^3 / D) * x * sum_{d}( dL/dy * w * x )
    The second term is the cross-channel coupling that distinguishes
    RMSNorm bwd from a per-channel scale.

(stage GS: gate * synapse * residual)
    From  x1 = x + syn * gate, gate = sigmoid(gate_pre):
        dL/dx (from this branch) = dL/dx1
        dL/dgated = dL/dx1
        dL/dsyn   = dL/dgated * gate
        dL/dgate  = dL/dgated * syn
        dL/dgate_pre = dL/dgate * gate * (1 - gate)
    Then GEMM through W_g (in glue) to get dL/d(spike_in)_via_gate
    and dL/d(W_g).

    From sparse synapse (in glue, masked GEMM):
        dL/d(spike_in)_via_syn = (W_syn * mask).T @ dL/dsyn
        dL/d(W_syn) = (dL/dsyn).T @ spike_in  *  mask

    Combined:  dL/d(spike_in) = via_gate + via_syn

(stage SP: SEW + spike + PLIF)
    spike_in = s + h_post  (SEW shortcut)
    dL/ds      = dL/d(spike_in)
    dL/dh_post += dL/d(spike_in)         (also receives carry from CfC scan)

    Surrogate ATan derivative on spike:
        ds/d(v_pre) = alpha / (2 * (1 + (pi/2 * alpha * (v_pre - thr))^2))
    From v_post = v_pre - s * thr:
        d(v_post)/d(v_pre) = 1 - thr * ds/d(v_pre)
        d(v_post)/d(thr)   = -s + (-thr) * (-ds/d(v_pre)) = ... see notes.
    More cleanly: use the chain rule with the saved spike + ds/d(v_pre).

(stage PLIF: membrane recurrence bwd)
    v_pre[t] = decay * v_post[t-1] + (1-decay) * h_post[t]
    d(v_pre)/d(h_post) = (1 - decay)
    d(v_pre)/d(v_post[t-1]) = decay
    d(v_pre)/d(decay) = v_post[t-1] - h_post[t]
    decay = exp(-1/exp(log_tau)),  d(decay)/d(log_tau) = decay * 1/exp(log_tau)

(stage CfC: scan bwd)
    h_pre[t] = A_t * h_post[t-1] + bvec
    h_post[t] = tanh(h_pre[t])
    A_t = exp(-delta * decay_a),  decay_a = exp(A_log)
    Reverse-time recurrence:
        dL/dh_pre[t] = dL/dh_post[t] * (1 - h_post^2)        (tanh')
        dL/dh_post[t-1] += dL/dh_pre[t] * A_t
        dL/dA_t  = dL/dh_pre[t] * h_post[t-1]
        dL/dbvec[t] = dL/dh_pre[t]
        dL/ddelta_t  = dL/dA_t * (-decay_a) * A_t  +  d_via_bvec
        dL/d(A_log)  = sum_t dL/dA_t * (-delta_t * decay_a) * A_t * (chain)
                     = -sum_t dL/dA_t * delta_t * A_t * decay_a / decay_a
                     -- actually simpler: A_t = exp(-delta * decay_a),
                        d(A_t)/d(decay_a) = -delta * A_t
                        d(decay_a)/d(A_log) = decay_a
                     => dL/d(A_log) = sum_t dL/dA_t * (-delta_t * A_t * decay_a)

(stage RN1: RMSNorm bwd)
    Symmetric to RN2 but operating on x (block input) and feeding back
    dL/dx through the scan via dL/dy1.

This file ships kernel implementations for the SCAN-backward (the only
non-elementwise reverse pass) plus elementwise fused tail kernels. The
glue layer chains them together with the cuBLAS GEMM gradients.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
HAS_TRITON: bool = False
try:  # pragma: no cover -- host-dependent
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if HAS_TRITON:

    PI_OVER_2: float = 1.5707963267948966

    @triton.jit
    def fused_hybrid_scan_bwd_kernel(
        # ---- forward state (read-only) ----
        delta_ptr,          # (B, T, D)   fp32
        bvec_ptr,           # (B, T, D)   fp32  (UNUSED in bwd, but keeps interface symmetry)
        A_log_ptr,          # (D,)        fp32
        log_tau_ptr,        # (D,)        fp32
        thr_ptr,            # (D,)        fp32
        h_pre_ptr,          # (B, T, D)   fp32 (saved by fwd)
        h_post_ptr,         # (B, T, D)   fp32 (saved by fwd)  -- in bwd we just use saved h_post
        v_pre_ptr,          # (B, T, D)   fp32 (saved)
        v_post_ptr,         # (B, T, D)   fp32 (saved)
        spike_ptr,          # (B, T, D)   fp32 (saved)
        h0_ptr,             # (B, D)      fp32 (initial CfC state, saved by glue)
        v0_ptr,             # (B, D)      fp32 (initial PLIF membrane, saved by glue)
        # ---- upstream gradients (read-only) ----
        gh_post_ptr,        # (B, T, D)   dL/dh_post -- accumulated from SEW + downstream
        gs_ptr,             # (B, T, D)   dL/dspike  -- from SEW+gate+syn
        # ---- output gradients (write) ----
        gdelta_ptr,         # (B, T, D)   dL/ddelta
        gbvec_ptr,          # (B, T, D)   dL/dbvec
        gA_log_ptr,         # (D,)        dL/d(A_log)            -- atomic
        glog_tau_ptr,       # (D,)        dL/d(log_tau)          -- atomic
        gthr_ptr,           # (D,)        dL/d(thr)              -- atomic
        gh0_ptr,            # (B, D)      dL/dh0                  -- write
        gv0_ptr,            # (B, D)      dL/dv0                  -- write
        # ---- strides ----
        d_b_str, d_t_str, d_d_str,
        sav_b_str, sav_t_str, sav_d_str,
        h0_b_str, h0_d_str,
        v0_b_str, v0_d_str,
        # ---- sizes & meta ----
        B, T, D,
        alpha,
        WRITE_GH0: tl.constexpr,
        WRITE_GV0: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Reverse-time scan that back-propagates through CfC + PLIF.

        Grid: (cdiv(D, BLOCK_D), B). Each program owns BLOCK_D channels of
        one batch element. The reverse loop walks t = T-1 .. 0 with the
        following carry state (registers, never spills):

            gh_carry  -- dL/dh_post[t-1] received from h_pre[t]'s scan
                          (CfC reverse).
            gv_carry  -- dL/dv_post[t-1] received from v_pre[t]'s recurrence
                          (PLIF reverse).
            gA_log_acc, glog_tau_acc, gthr_acc -- per-channel
                          accumulators that get atomically added at the end.
        """
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        # ---- per-channel constants ----
        A_log = tl.load(A_log_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
        decay_a = tl.exp(A_log)
        log_tau = tl.load(log_tau_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
        plif_decay = tl.exp(-1.0 / tl.exp(log_tau))
        plif_one_minus_decay = 1.0 - plif_decay
        # d(plif_decay) / d(log_tau)
        #   plif_decay = exp(-1/tau), tau = exp(log_tau)
        #   d/d(log_tau) plif_decay = exp(-1/tau) * 1/tau^2 * tau
        #                            = plif_decay * 1 / exp(log_tau)
        d_decay_d_logtau = plif_decay / tl.exp(log_tau)

        thr = tl.load(thr_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)

        # ---- per-channel accumulators ----
        gA_log_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        glog_tau_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        gthr_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # ---- carries ----
        gh_carry = tl.zeros([BLOCK_D], dtype=tl.float32)
        gv_carry = tl.zeros([BLOCK_D], dtype=tl.float32)

        # ---- reverse loop ----
        for t in range(T - 1, -1, -1):
            in_off = pid_b * d_b_str + t * d_t_str + d_off * d_d_str
            sav_off = pid_b * sav_b_str + t * sav_t_str + d_off * sav_d_str

            delta_t = tl.load(delta_ptr + in_off, mask=d_mask, other=0.0).to(tl.float32)
            h_pre_t = tl.load(h_pre_ptr + sav_off, mask=d_mask, other=0.0).to(tl.float32)
            h_post_t = tl.load(h_post_ptr + sav_off, mask=d_mask, other=0.0).to(tl.float32)
            v_pre_t = tl.load(v_pre_ptr + sav_off, mask=d_mask, other=0.0).to(tl.float32)
            spike_t = tl.load(spike_ptr + sav_off, mask=d_mask, other=0.0).to(tl.float32)

            gh_post_t = tl.load(gh_post_ptr + in_off, mask=d_mask, other=0.0).to(tl.float32)
            gs_t = tl.load(gs_ptr + in_off, mask=d_mask, other=0.0).to(tl.float32)

            # ---- ATan surrogate derivative (saved m = v_pre - thr) ----
            m_t = v_pre_t - thr
            x = alpha * m_t
            denom = 1.0 + (PI_OVER_2 * x) * (PI_OVER_2 * x)
            ds_dm = alpha / (2.0 * denom)
            # ds/d(v_pre) = ds_dm,  ds/d(thr) = -ds_dm

            # ---- back through PLIF reset: v_post = v_pre - s * thr ----
            #     dL/dv_pre   = dL/dv_post * 1   (path through v_post)
            #                 + dL/dv_post * (-thr) * ds_dm   (path through s)
            #                 + dL/ds * ds_dm                 (path through ds)
            # plus: dL/dh_post += dL/ds * 1  (SEW shortcut, ALREADY folded
            # into gh_post_ptr by the glue layer before this kernel runs).
            # Receive gv_post from carry and gs from upstream.
            gv_post_total = gv_carry  # dL/dv_post[t]  (only the carry; v_post[t]
            # is consumed by the t+1 PLIF step which already populated gv_carry)
            # gs_t comes from the SEW + gate + synapse path (glue).

            # gradient through reset to v_pre and thr
            ds_term = gs_t * ds_dm  # contribution to dL/dv_pre via s itself
            # dL/dv_pre from v_post path:
            gv_pre = gv_post_total * (1.0 - thr * ds_dm)
            # plus gradient via direct s path:
            gv_pre += ds_term
            # threshold gradient:
            #    dL/dthr += -gv_post_total * (s + thr * (-ds_dm)*(-1))   ... let's redo
            # Cleaner: thr appears in v_post = v_pre - s*thr.  d(v_post)/d(thr)
            # = -s - thr * d(s)/d(thr) (since s depends on thr through Heaviside
            # surrogate: ds/d(thr) = -ds_dm).
            #   d(v_post)/d(thr) = -s + thr * ds_dm
            # So:
            gthr_acc += gv_post_total * (-spike_t + thr * ds_dm)
            # Plus thr also appears in ds itself (s = H(v_pre - thr)):
            gthr_acc += gs_t * (-ds_dm)

            # ---- back through PLIF integrator:
            #     v_pre = plif_decay * v_post[t-1] + (1-plif_decay) * h_post
            # ----
            gh_post_via_plif = gv_pre * plif_one_minus_decay
            gv_carry_next = gv_pre * plif_decay  # dL/dv_post[t-1]
            # log_tau gradient:
            #     d(v_pre)/d(plif_decay) = v_post[t-1] - h_post
            # need v_post[t-1] = (saved at t-1) -- load it.
            if t > 0:
                v_post_tm1 = tl.load(
                    v_post_ptr + pid_b * sav_b_str + (t - 1) * sav_t_str + d_off * sav_d_str,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
            else:
                v_post_tm1 = tl.load(
                    v0_ptr + pid_b * v0_b_str + d_off * v0_d_str,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
            d_vpre_d_decay = v_post_tm1 - h_post_t
            glog_tau_acc += gv_pre * d_vpre_d_decay * d_decay_d_logtau

            # ---- accumulate dL/dh_post[t] ----
            # Sources:
            #   (a) gh_post_t -- from upstream (CfC's own consumer + SEW
            #       which the glue folded into gh_post_ptr).
            #   (b) gh_post_via_plif -- backward through PLIF
            #       integrator.
            #   (c) carry from CfC scan (h_post[t] = h_post[t], the
            #       previous reverse step's gh_carry comes in as
            #       dL/dh_post[t-1] for the NEXT iteration; not added
            #       here.  But the carry from the PRIOR reverse step
            #       (i.e. the next-time-step) already came in as
            #       gh_post_ptr (NO -- glue is responsible for SEW
            #       forwarding; carry is for the within-scan recurrence.
            gh_post_total = gh_post_t + gh_post_via_plif + gh_carry

            # ---- back through h_post = tanh(h_pre): ----
            #     dL/dh_pre = dL/dh_post * (1 - h_post^2)
            tanh_d = 1.0 - h_post_t * h_post_t
            gh_pre = gh_post_total * tanh_d

            # ---- back through h_pre = A_t * h_post[t-1] + bvec: ----
            # need h_post[t-1] -- saved or h0
            if t > 0:
                h_post_tm1 = tl.load(
                    h_post_ptr + pid_b * sav_b_str + (t - 1) * sav_t_str + d_off * sav_d_str,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
            else:
                h_post_tm1 = tl.load(
                    h0_ptr + pid_b * h0_b_str + d_off * h0_d_str,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)

            # A_t at this step
            A_t = tl.exp(-delta_t * decay_a)

            # Gradients on the bilinear scan:
            #   dL/dA_t        = gh_pre * h_post[t-1]
            #   dL/dh_post[t-1]+= gh_pre * A_t       (carry)
            #   dL/dbvec       = gh_pre
            gA_t = gh_pre * h_post_tm1
            gh_carry_next = gh_pre * A_t
            tl.store(gbvec_ptr + in_off, gh_pre, mask=d_mask)
            # dL/ddelta_t = gA_t * d(A_t)/d(delta) = gA_t * (-decay_a) * A_t
            gdelta_t = gA_t * (-decay_a * A_t)
            tl.store(gdelta_ptr + in_off, gdelta_t, mask=d_mask)
            # dL/d(A_log) per channel:
            #   d(A_t)/d(decay_a) = -delta * A_t
            #   d(decay_a)/d(A_log) = decay_a
            # => dL/d(A_log) += gA_t * (-delta_t * A_t * decay_a)
            gA_log_acc += gA_t * (-delta_t * A_t * decay_a)

            # advance carries
            gh_carry = gh_carry_next
            gv_carry = gv_carry_next

        # ---- write per-batch h0 / v0 grads ----
        if WRITE_GH0:
            tl.store(
                gh0_ptr + pid_b * h0_b_str + d_off * h0_d_str,
                gh_carry, mask=d_mask,
            )
        if WRITE_GV0:
            tl.store(
                gv0_ptr + pid_b * v0_b_str + d_off * v0_d_str,
                gv_carry, mask=d_mask,
            )

        # ---- atomic-add per-channel grads ----
        tl.atomic_add(gA_log_ptr + d_off, gA_log_acc, mask=d_mask)
        tl.atomic_add(glog_tau_ptr + d_off, glog_tau_acc, mask=d_mask)
        tl.atomic_add(gthr_ptr + d_off, gthr_acc, mask=d_mask)

    @triton.jit
    def fused_rmsnorm_bwd_kernel(
        # ---- inputs ----
        gy_ptr,             # (B, T, D)  dL/dy
        x_ptr,              # (B, T, D)  forward input
        w_ptr,              # (D,)       weight
        rstd_ptr,           # (B, T)     saved 1/sqrt(mean(x^2)+eps)
        # ---- outputs ----
        gx_ptr,             # (B, T, D)  dL/dx
        gw_ptr,             # (D,)       dL/dw  -- atomic_add per (b,t) tile
        # ---- strides ----
        x_b_str, x_t_str, x_d_str,
        w_str,
        rstd_b_str, rstd_t_str,
        # ---- sizes & meta ----
        B, T, D,
        BLOCK_D: tl.constexpr,
    ):
        """Closed-form RMSNorm backward.

        For y = x * rstd * w,  rstd = 1/sqrt(mean(x^2) + eps):

            dL/dw = sum_{b,t} dL/dy * x * rstd
            dL/dx = (rstd * w) * dL/dy
                  - (rstd^3 / D) * x * sum_d ( dL/dy * w * x )

        Grid: (T, B). Each program owns ALL D channels of one (b, t)
        row -- needed for the cross-channel sum_d term.
        """
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        base = pid_b * x_b_str + pid_t * x_t_str + d_off * x_d_str
        gy = tl.load(gy_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + d_off * w_str, mask=d_mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + pid_b * rstd_b_str + pid_t * rstd_t_str)

        # gw contribution from this (b,t) -- atomically added.
        gw_local = gy * x * rstd
        tl.atomic_add(gw_ptr + d_off * w_str, gw_local, mask=d_mask)

        # Cross-channel sum_d ( gy * w * x )
        gywx = gy * w * x
        s = tl.sum(gywx, axis=0)
        # gx = (rstd * w) * gy - (rstd^3 / D) * x * s
        gx = rstd * w * gy - (rstd * rstd * rstd / D) * x * s
        tl.store(gx_ptr + base, gx.to(gx_ptr.dtype.element_ty), mask=d_mask)

    @triton.jit
    def fused_swiglu_bwd_kernel(
        # ---- inputs ----
        g_ffn_act_ptr,      # (B, T, H)  dL/dffn_act = dL/d(silu(g_pre)*up_pre)
        g_pre_ptr,          # (B, T, H)  forward gate pre-activation
        up_pre_ptr,         # (B, T, H)  forward up pre-activation
        # ---- outputs ----
        g_g_pre_ptr,        # (B, T, H)  dL/d(g_pre)
        g_up_pre_ptr,       # (B, T, H)  dL/d(up_pre)
        # ---- strides ----
        h_b_str, h_t_str, h_d_str,
        # ---- sizes & meta ----
        B, T, H,
        BLOCK_H: tl.constexpr,
    ):
        """Closed-form SwiGLU backward.

        For y = silu(g) * up = g * sigmoid(g) * up:
            dy/dup = silu(g)
            dy/dg  = sigmoid(g) * up + g * sigmoid(g) * (1 - sigmoid(g)) * up
                   = sigmoid(g) * up * (1 + g * (1 - sigmoid(g)))

        Grid: (T, B). Tile across H with BLOCK_H.
        """
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)

        h_off = tl.arange(0, BLOCK_H)
        h_mask = h_off < H
        base = pid_b * h_b_str + pid_t * h_t_str + h_off * h_d_str

        g_act = tl.load(g_ffn_act_ptr + base, mask=h_mask, other=0.0).to(tl.float32)
        g_pre = tl.load(g_pre_ptr + base, mask=h_mask, other=0.0).to(tl.float32)
        up = tl.load(up_pre_ptr + base, mask=h_mask, other=0.0).to(tl.float32)

        sig = tl.where(
            g_pre >= 0.0,
            1.0 / (1.0 + tl.exp(-g_pre)),
            tl.exp(g_pre) / (1.0 + tl.exp(g_pre)),
        )
        silu = g_pre * sig
        # dL/dup = g_act * silu
        gup = g_act * silu
        # dL/dg_pre = g_act * sig * up * (1 + g_pre * (1 - sig))
        ggp = g_act * sig * up * (1.0 + g_pre * (1.0 - sig))
        tl.store(g_g_pre_ptr + base, ggp.to(g_g_pre_ptr.dtype.element_ty), mask=h_mask)
        tl.store(g_up_pre_ptr + base, gup.to(g_up_pre_ptr.dtype.element_ty), mask=h_mask)


# ---------------------------------------------------------------------------
if not HAS_TRITON:  # pragma: no cover
    fused_hybrid_scan_bwd_kernel = None  # type: ignore[assignment]
    fused_rmsnorm_bwd_kernel = None  # type: ignore[assignment]
    fused_swiglu_bwd_kernel = None  # type: ignore[assignment]


__all__ = [
    "HAS_TRITON",
    "fused_hybrid_scan_bwd_kernel",
    "fused_rmsnorm_bwd_kernel",
    "fused_swiglu_bwd_kernel",
]
