#!/usr/bin/env bash
# download_sft_extended.sh -- fetch extended SFT corpora beyond alpaca.
#
# Pulls 7 SFT corpora from hf-mirror (primary) + modelscope (secondary):
#   sharegpt_zh    real Chinese chat from xitao/sharegpt_zh (~90K)
#   coig           Chinese Open Instruction Generalist (~191K)
#   oa_zh          OpenAssistant translated Chinese conversations (~40K)
#   wizard_zh      WizardLM-zh evol-instruct in Chinese
#   gsm8k_cot      GSM8K math CoT chains
#   codealpaca     CodeAlpaca 20K coding instructions
#   self_instruct  200 high-quality self-bootstrap seeds
#
# Per-corpus jq parse check + min-entries verify.
# Defensive: missing corpus falls back to a synthetic 100-example placeholder.
#
# Usage:
#   bash scripts/download_sft_extended.sh                     # all 7
#   bash scripts/download_sft_extended.sh --only sharegpt_zh  # one
#   bash scripts/download_sft_extended.sh --size smoke        # 100/each
#   bash scripts/download_sft_extended.sh --size small        # 5K each
#   bash scripts/download_sft_extended.sh --size full         # default

set -u
set -o pipefail

OUT_DIR="${OUT_DIR:-/workspace/data/sft}"
MIN_BYTES="${MIN_BYTES:-65536}"    # 64KB floor (self_instruct seed_tasks ~110KB)
TIMEOUT="${TIMEOUT:-180}"
ONLY=""
SIZE="full"

usage() {
  sed -n '2,21p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage ;;
    --only)    ONLY="$2"; shift 2 ;;
    --out)     OUT_DIR="$2"; shift 2 ;;
    --size)    SIZE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$SIZE" in
  smoke|small|full) ;;
  *) echo "bad --size: $SIZE (need smoke|small|full)" >&2; exit 2 ;;
esac

# Per-corpus: <name>|<dest_subdir>|<min_entries>|<url1>|<url2>
# url1=hf-mirror, url2=modelscope (or fallback github raw)
MIRRORS=(
  "sharegpt_zh|sharegpt_zh|50000|https://hf-mirror.com/datasets/xitao/sharegpt_zh/resolve/main/sharegpt_zh.json|https://www.modelscope.cn/api/v1/datasets/xitao/sharegpt_zh/repo?Revision=master&FilePath=sharegpt_zh.json"
  "coig|coig|150000|https://hf-mirror.com/datasets/BAAI/COIG/resolve/main/train.json|https://www.modelscope.cn/api/v1/datasets/BAAI/COIG/repo?Revision=master&FilePath=train.json"
  "oa_zh|oa_zh|30000|https://hf-mirror.com/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz|https://hf-mirror.com/datasets/sunzeyeah/chinese_chatgpt_corpus/resolve/main/oasst_zh.json"
  "wizard_zh|wizard_zh|50000|https://hf-mirror.com/datasets/silk-road/wizard_vicuna_70k_chinese/resolve/main/wizard_vicuna_70k_chinese.json|https://hf-mirror.com/datasets/wizardlm-zh/wizardlm_70k_chinese/resolve/main/data.json"
  "gsm8k_cot|gsm8k_cot|7000|https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet|https://hf-mirror.com/datasets/qwedsacf/grade-school-math-instructions/resolve/main/data.json"
  "codealpaca|codealpaca|18000|https://hf-mirror.com/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json|https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
  "self_instruct|self_instruct|175|https://hf-mirror.com/datasets/yizhongw/self_instruct/resolve/main/seed_tasks.jsonl|https://raw.githubusercontent.com/yizhongw/self-instruct/main/data/seed_tasks.jsonl"
)

mkdir -p "$OUT_DIR"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

apply_size_cap() {
  # cap entries by --size for smoke/small modes
  local path="$1" size="$2"
  local cap=""
  case "$size" in
    smoke) cap=100 ;;
    small) cap=5000 ;;
    full)  return 0 ;;
  esac
  if [[ -z "$cap" ]]; then return 0; fi
  if ! command -v jq >/dev/null 2>&1; then
    warn "jq not available, cannot apply size cap on $path"
    return 0
  fi
  local tmp="$path.cap.tmp"
  if jq -e 'type == "array"' "$path" >/dev/null 2>&1; then
    jq ".[0:$cap]" "$path" > "$tmp" && mv -f "$tmp" "$path"
    log "capped $path to $cap entries"
  fi
}

# Validate JSON or JSONL; for parquet just verify size.
verify_file() {
  local path="$1" min_entries="$2"
  [[ -f "$path" ]] || return 1
  local size; size=$(wc -c < "$path" 2>/dev/null || echo 0)
  if (( size < MIN_BYTES )); then
    warn "$path too small ($size B < $MIN_BYTES B)"
    return 1
  fi
  if [[ "$path" == *.parquet || "$path" == *.gz ]]; then
    return 0  # binary; skip jq
  fi
  if command -v jq >/dev/null 2>&1; then
    if head -c 65536 "$path" | jq -e 'type == "array"' >/dev/null 2>&1; then
      local n; n=$(jq 'length' "$path" 2>/dev/null || echo 0)
      if (( n < min_entries )); then
        warn "$path has $n entries (< $min_entries expected)"
        return 1
      fi
    elif head -n 5 "$path" | jq -e . >/dev/null 2>&1; then
      local n; n=$(wc -l < "$path" 2>/dev/null || echo 0)
      if (( n < min_entries )); then
        warn "$path has $n lines (< $min_entries expected)"
        return 1
      fi
    else
      warn "$path: not parseable as JSON array or JSONL"
      return 1
    fi
  fi
  return 0
}

# Generate a synthetic 100-example placeholder when all mirrors fail.
write_synthetic_placeholder() {
  local name="$1" path="$2"
  log "$name: writing synthetic 100-example placeholder to $path"
  local py
  if command -v python3 >/dev/null 2>&1 && python3 -c '' 2>/dev/null; then
    py=python3
  elif command -v python >/dev/null 2>&1 && python -c '' 2>/dev/null; then
    py=python
  else
    warn "$name: no python available; cannot write placeholder"
    return 1
  fi
  "$py" - "$path" "$name" <<'PY'
import json, sys, hashlib
out_path, name = sys.argv[1], sys.argv[2]
templates = {
    "sharegpt_zh": (
        "你好，能帮我{}吗？",
        "当然可以。这里是关于{}的简要说明：首先...",
    ),
    "coig":       ("请解释什么是{}。", "{}是一个常见的概念，主要含义是..."),
    "oa_zh":      ("我想了解{}相关的知识", "好的。{}涉及以下几个方面：1)..."),
    "wizard_zh":  ("写一段关于{}的代码", "下面是一个示例：\n```python\n# {} 示例\n```"),
    "gsm8k_cot":  ("一个班里有 {} 名学生，每人有 3 本书，总共多少本？",
                   "我们一步一步算：每人 3 本，共 {} 人，所以 3 × {} = 答案"),
    "codealpaca": ("Write a Python function to {}.",
                   "Here is a clean implementation:\n\n```python\ndef solve():\n    # {}\n    pass\n```"),
    "self_instruct": ("Generate a high-quality instruction about {}.",
                      "Instruction: Explain the topic of {} in three paragraphs."),
}
topic_pool = ["machine learning", "排序算法", "数据库索引", "光合作用", "金融建模",
              "宇宙起源", "中医基础", "并发编程", "图论", "文学创作"]
qa, ans = templates.get(name, ("Question about {}.", "Answer about {}."))
items = []
for i in range(100):
    topic = topic_pool[i % len(topic_pool)]
    items.append({
        "instruction": qa.format(topic),
        "input": "",
        "output": ans.format(topic, i + 1, i + 1) if "{}" in ans else ans,
        "source": f"{name}_synthetic",
    })
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=1)
print(f"[synthetic] wrote {len(items)} placeholder examples to {out_path}")
PY
}

fetch_one() {
  local name="$1" subdir="$2" min_entries="$3"; shift 3
  local urls=("$@")
  local target_dir="$OUT_DIR/$subdir"
  mkdir -p "$target_dir"
  local out="$target_dir/train.json"

  if verify_file "$out" "$min_entries"; then
    log "$name: already present and verified, skip"
    return 0
  fi

  for url in "${urls[@]}"; do
    log "$name: trying $url"
    if curl -fL --connect-timeout 20 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$url" 2>/dev/null; then
      mv -f "$out.tmp" "$out"
      if verify_file "$out" "$min_entries"; then
        log "$name: OK ($(wc -c < "$out") B from $url)"
        apply_size_cap "$out" "$SIZE"
        return 0
      fi
    fi
    rm -f "$out.tmp"
    warn "$name: failed $url"
  done
  warn "$name: all mirrors failed; writing synthetic placeholder"
  write_synthetic_placeholder "$name" "$out"
  return 1
}

ok=0; fail=0; placeholder=0
for entry in "${MIRRORS[@]}"; do
  IFS='|' read -r name subdir min_entries u1 u2 <<<"$entry"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then continue; fi
  if fetch_one "$name" "$subdir" "$min_entries" "$u1" "$u2"; then
    ok=$((ok+1))
  else
    placeholder=$((placeholder+1))
  fi
done

log "done: $ok real, $placeholder placeholder; out=$OUT_DIR size=$SIZE"
[[ "$placeholder" -eq 0 ]]
