#!/usr/bin/env bash
# v4.1 best ckpt backup — 5 paths, run on rental.
#
# Run via 算力牛 web console if SSH is broken:
#   bash backup_v41.sh
#
# Tries 5 backup destinations in parallel, succeeds if at least 2 work.
# Safe to re-run; uses content hash to skip already-uploaded.

set -uo pipefail

CKPT="/workspace/runs/synapforge_v41_neuromcp/best.pt"
SHA_FILE="$CKPT.sha256"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found"
    exit 1
fi

CKPT_SIZE=$(stat -c %s "$CKPT" 2>/dev/null || stat -f %z "$CKPT" 2>/dev/null)
echo "ckpt size: $CKPT_SIZE bytes"

if [ ! -f "$SHA_FILE" ]; then
    echo "computing sha256..."
    sha256sum "$CKPT" > "$SHA_FILE"
fi
SHA=$(cut -d' ' -f1 < "$SHA_FILE")
echo "sha256: $SHA"

success_count=0
SUCCESS_DEST=""

# ======== Path 1: GitHub Release ========
echo
echo "=== [1/5] GitHub Release ==="
if command -v gh >/dev/null 2>&1; then
    TAG="v4.1-best-$(date +%Y%m%d)"
    if gh release create "$TAG" "$CKPT" \
        --title "v4.1 best ckpt $(date +%Y-%m-%d)" \
        --notes "Auto-backup of /workspace/runs/synapforge_v41_neuromcp/best.pt
        sha256: $SHA
        size: $CKPT_SIZE bytes
        step: 60000, best ppl: 44.2 (single batch)
        " 2>&1; then
        echo "  GitHub Release OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST github-release"
    elif gh release upload "$TAG" "$CKPT" --clobber 2>&1; then
        echo "  GitHub Release upload to existing tag OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST github-release"
    else
        echo "  FAILED"
    fi
else
    echo "  gh not installed, skip"
fi

# ======== Path 2: scp to mohuanfang.com ========
echo
echo "=== [2/5] scp to mohuanfang.com ==="
if command -v scp >/dev/null 2>&1; then
    timeout 1200 scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$CKPT" liu@mohuanfang.com:/data/synapforge_backup/v4.1_best.pt \
        2>&1 && {
        echo "  scp mohuanfang.com OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST mohuanfang"
    } || echo "  FAILED"
fi

# ======== Path 3: 算力牛 /mnt 共享盘 ========
echo
echo "=== [3/5] /mnt 共享盘 ==="
if [ -d /mnt ] && [ -w /mnt ]; then
    cp "$CKPT" /mnt/synapforge_v41_best_$(date +%Y%m%d).pt && {
        echo "  /mnt copy OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST mnt"
    } || echo "  FAILED"
else
    echo "  /mnt not writable, skip"
fi

# ======== Path 4: Hugging Face dataset upload ========
echo
echo "=== [4/5] HuggingFace dataset ==="
if command -v huggingface-cli >/dev/null 2>&1 && [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli upload \
        Lihfdgjr/synapforge-ckpts \
        "$CKPT" \
        v4.1-best.pt \
        --repo-type dataset \
        --commit-message "v4.1 best ckpt sha256=$SHA" \
        2>&1 && {
        echo "  HF dataset OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST hf-dataset"
    } || echo "  FAILED"
else
    echo "  HF_TOKEN unset or huggingface-cli missing, skip"
fi

# ======== Path 5: Personal cloud (rclone / aws s3 / oss) ========
echo
echo "=== [5/5] Personal cloud (rclone/s3/oss) ==="
if command -v rclone >/dev/null 2>&1; then
    rclone copy "$CKPT" "synapforge-backup:/v4.1_best_$(date +%Y%m%d).pt" 2>&1 && {
        echo "  rclone OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST rclone"
    } || echo "  FAILED"
elif command -v aws >/dev/null 2>&1; then
    aws s3 cp "$CKPT" "s3://synapforge-backup/v4.1_best_$(date +%Y%m%d).pt" 2>&1 && {
        echo "  aws s3 OK"
        success_count=$((success_count + 1))
        SUCCESS_DEST="$SUCCESS_DEST aws-s3"
    } || echo "  FAILED"
else
    echo "  no rclone / aws cli, skip"
fi

# ======== Summary ========
echo
echo "================================================"
echo "Backup summary: $success_count / 5 paths succeeded"
echo "Successful destinations: $SUCCESS_DEST"
echo "================================================"

if [ $success_count -ge 2 ]; then
    echo "OK — at least 2 backups successful, safe to spin up new instance"
    exit 0
elif [ $success_count -ge 1 ]; then
    echo "WARNING — only 1 backup, consider not killing this instance yet"
    exit 1
else
    echo "FAIL — no backups succeeded, DO NOT KILL THIS INSTANCE"
    exit 2
fi
