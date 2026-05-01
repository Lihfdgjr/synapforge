#!/bin/bash
# Background loop: keep most recent 5 step_*.pt + all best_*.pt +
# all phase_change_*.pt. Runs every 5 minutes.
#
# Usage: setsid bash -c 'scripts/ckpt_cleanup.sh DIR' </dev/null & disown
# Without DIR arg, defaults to v24h_qwen3 (current canonical run, 2026-05-01).
#
# Why we need this: trainer SAVE_EVERY=2000 + 1.83GB ckpts = ~50GB/24h.
# Rental system disk is 100GB. Without rotation, 99.9% full → torch.save
# crashes (see scripts/launch_v24h_qwen3.sh header for the original incident).
DIR="${1:-/workspace/runs/v24h_qwen3}"
while true; do
  cd "$DIR" 2>/dev/null || { sleep 60; continue; }
  ls -t step_[0-9]*.pt 2>/dev/null | tail -n +6 | xargs -r rm -f
  sleep 300
done
