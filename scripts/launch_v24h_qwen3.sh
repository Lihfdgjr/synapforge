#!/bin/bash
# Launch v24h_qwen3 trainer (PLIF-reinit warmstart, bs=64 to fit z-loss in 80GB).
#
# 2026-05-01 incident: v24h_qwen Run 2 hit "system disk 99.9% full" because
# SAVE_EVERY=250 + 1.83GB ckpts → 53 ckpts × 1.83GB = 97GB. Trainer crashed
# when torch.save couldn't write. Recovery:
#   1. Manual ckpt cleanup (kept 9, freed 77GB).
#   2. SAVE_EVERY raised to 2000.
#   3. ckpt_cleanup.sh background loop keeps last 5 step_* + all best_* +
#      all phase_change_*.
#   4. PLIF reinit (scripts/reinit_plif.py) on step_002250.pt — but PLIFCell
#      param naming changed (A_log → log_tau), so old PLIF weights are
#      orphaned anyway. Net effect: warmstart with random PLIF init at
#      patched defaults (threshold=0.05, tau=2.5).
#   5. Batch 128 → 64 because new z-loss path materializes (B*T, V) =
#      18.55GB tensor for logsumexp; bs=64 cuts it to ~9.5GB.
#
# Use systemd-run --user OR setsid+disown to survive MCP shell closure.
set -e
cd /workspace/synapforge_git
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
exec python3 -u train_100m_kd.py \
  --warmstart /workspace/runs/v24h_qwen/step_002250_plif_reinit.pt \
  --teacher Qwen/Qwen2.5-0.5B \
  --teacher-fallback-ckpt "" \
  --backend triton_block \
  --batch-size 64 \
  --kd-every 4 \
  --steps 30000 \
  --out /workspace/runs/v24h_qwen3 \
  --phase-aware \
  > /workspace/runs/v24h_qwen3/train.log 2>&1
