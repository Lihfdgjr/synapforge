# Quickstart

Three paths, in order of how much time you want to spend.

---

## Path 1 — 30 seconds, see it

The fastest way to verify SynapForge is real. CPU only. No GPU, no
API keys, no checkpoints.

```bash
pip install synapforge
synapforge-demo all
```

**Expected runtime**: 30-60s on a 2020-era laptop.

**Expected output (abbreviated)**:

```
SynapForge — 30 second pitch
  GPT-class transformers cost millions to train. We built a 375-million-
  parameter LNN+SNN hybrid: continuous-time CfC + spiking PLIF + Hebbian ...

=== NeuroMCP 4-button demo ===
  agent: 73.0K params
  initial synapse density: 5.0%
  initial codebook K: 9
  trial   0  hit_rate=0.31  density=5.0%  K= 9  loss=0.521
  trial  16  hit_rate=0.62  density=8.7%  K=10  loss=0.341  GREW
  ...
  trial  79  hit_rate=1.00  density=27.9% K=14  loss=0.087
  density: 5.0% -> 27.9%
  zero <tool_call> tokens emitted.

=== R-fold bench ===
  device: cpu
  correctness: R=1 rel_err=1.5e-06   R=8 rel_err=3.2e-03
  ...
  CPU: fold loses for N>=128 (LAPACK solve overhead).
  Real win is GPU + N>=256.

=== STDP self-organization demo ===
  trial  50  density= 0.7%  mean|W|=0.0077  spike_rate=0.22
  trial 100  density= 8.1%  mean|W|=0.0182  spike_rate=0.23
  trial 199  density=27.3%  mean|W|=0.0390  spike_rate=0.09
  no optimizer, no loss. structure emerged from the Hebbian rule alone.

=== Chat demo (5 EN + 5 ZH) ===
  ckpt unavailable, replaying recorded transcript:
  > Once upon a time
       there was a young girl who lived in a small village near the forest...
  > 中国的首都是
      北京。北京是中国的政治、文化和国际交流中心...
```

**Common errors**:

- `ModuleNotFoundError: synapforge` — installed in a different venv.
  Fix: `which python; pip install --user synapforge`.
- `RuntimeError: torch not built with CUDA` — harmless on CPU; the
  demos detect and run on CPU.
- Garbled Chinese in console output (Windows `cp936` / `gbk`) — the
  saved `chat_demo.json` is correct UTF-8; only the live console
  rendering is mangled. Open the JSON in any modern editor.

---

## Path 2 — Train it yourself

Spin up your own checkpoint. Needs 1× A100 / A800 80GB or equivalent.
~24h to chat-usable (ppl ~50), ~7 days to chat-fluent (ppl ~25).

### Setup

```bash
git clone https://github.com/Lihfdgjr/synapforge
cd synapforge
git checkout v4.2-staging
pip install -e .
```

### Pre-download data + teacher

```bash
python -m synapforge.data.fetch_datasets \
  --out /workspace/data \
  --include fineweb-en,alpaca-zh,alpaca-en,gsm8k,agent_math_gold

# Qwen2.5-0.5B as KD teacher
huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir /workspace/teachers/qwen2.5-0.5b
```

**Expected runtime**: ~30-60 min for ~60GB of data.

### Launch trainer

```bash
python train_100m_kd.py \
  --warmstart /workspace/runs/synapforge_v41/best.pt \
  --out-dir /workspace/runs/v42 \
  --teacher-path /workspace/teachers/qwen2.5-0.5b \
  --steps 60000 --batch-size 8 --seq-len 1024 --grad-accum 2 \
  --lr 2e-4 --kd-weight 0.3
```

**Expected runtime + output**:

```
[step 0]    ce=4.05  ppl=57.4   tok/s=22340   (warmstart honored)
[step 100]  ce=3.82  ppl=45.6   tok/s=21800
[step 5000] ce=3.50  ppl=33.1   tok/s=22100
[step 30000] ce=3.30  ppl=27.1  (5 GPU-h)
[step 60000] ce=3.22  ppl=25.0  (10 GPU-h)
```

**Common errors**:

- `Path("Qwen/...").exists() == False` (kd_weight=0 silently) — old
  bug; ensure `--teacher-path` is a local dir, not a HF repo id.
  Fixed in v4.2-staging.
- `OOM at z-loss step` — bs × seq × vocab > 8GB. Drop `--batch-size`
  to 4 or `--seq-len` to 1024.
- `tok/s drops to 1100` (~17× slow) — PerDomainNeuroMCP runs all 4
  heads. Use `--neuromcp-routing top1` (planned flag in v4.3).
- `KD teacher download flake` — HF Hub flake. Retry with
  `huggingface-cli download` directly; trainer accepts the local dir.
- `cold-start ppl very high` (>1000) — first 50-100 steps with no
  warmstart will look broken. Wait. By step 500 it should be in the
  ~100s range.

### Watch it train

```bash
tail -f /workspace/runs/v42/train.log
```

Look for: ppl monotonically dropping, `spike_rate` between 0.05 and
0.30 (PLIF healthy), `density` of NeuroMCP rising past 10%.

---

## Path 3 — Chat with it

Once you have a checkpoint (yours from Path 2 or one we share):

### Option A — packaged CLI

```bash
synapforge-demo chat \
  --ckpt /workspace/runs/v42/best.pt \
  --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
  --save my_chat.json
```

**Expected runtime**: ~30-60s for 10 prompts on CPU; ~5-10s on GPU.

### Option B — interactive REPL

```bash
python scripts/chat_repl.py \
  --ckpt /workspace/runs/v42/best.pt \
  --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
  --temperature 0.7 \
  --max-new 80 \
  --save chat_session.json
```

Type a prompt at the `>` prompt; Ctrl-D / Ctrl-C exits. All
(prompt, response) pairs are appended to `--save` so the transcript
survives an SSH drop.

**Expected output**:

```
> What is 2+2?
2+2 equals 4.
  (1.3s)

> 你好
你好!很高兴见到你。
  (1.1s)
```

**Common errors**:

- `WARN: ckpt ... not found, using RANDOM INIT` — output will be
  garbage tokens. Verify `--ckpt` path with `ls -la`.
- `Generation produces only "###"` — instruction template mismatch.
  The model expects `### Instruction:\n...\n### Response:\n` exactly
  (no trailing newline). Use the REPL or `chat_demo.py`, both wrap
  prompts correctly.
- `<generation error: max_seq exceeded>` — prompt + max-new > 2048.
  Drop `--max-new` to 40 or shorten the prompt.
- `imported torch but no CUDA` — REPL defaults to `--device cuda`. On
  CPU, pass `--device cpu`.

---

## Top 5 known issues + workarounds

### 1. GH Release 100MB cap blocks ckpt upload

**Symptom**: `gh release upload` fails with HTTP 422 on files >100MB.

**Workaround**: split with `split -b 95M ckpt.pt ckpt.pt.part_` and
upload parts. `auto_ckpt_backup.py` already does this. `mohuanfang.com`
backup has no cap and gets the full ckpt; GH Release is a redundant
copy.

### 2. OOM at bs=128 + Qwen vocab

**Symptom**: `torch.cuda.OutOfMemoryError` at the z-loss step on a
single 80GB card.

**Why**: 128 × 1024 × 151643 × 4 bytes = 80GB peak in z-loss.

**Workaround**: enable gradient checkpointing
(`--grad-checkpoint`) and drop to `--batch-size 64 --grad-accum 2`.
Same effective batch size, fits in 60GB peak.

### 3. KD teacher download flake

**Symptom**: `huggingface_hub.utils._errors.RepositoryNotFoundError`
on first launch even though the repo id is right.

**Why**: HF Hub rate-limits unauthenticated transit during regional
peak hours.

**Workaround**: `export HF_TOKEN=hf_...` (free token) and retry. Or
download once with the CLI and pass `--teacher-path` as a local dir.

### 4. Cold-start ppl very high

**Symptom**: First 50-100 steps log ppl in the 5000-50000 range.
Looks like the model is broken.

**Why**: Random init + sparse synapse layer (95% sparsity) takes
~50 steps for the dense FFN to dominate the loss before the synapse
kicks in.

**Workaround**: don't panic. By step 500 it should be <500 ppl. By
step 5000 it should be <50. If it's still >1000 after step 1000,
something is genuinely wrong — check `--lr` (try 2e-4) and verify
warmstart loaded (look for "loaded N missing keys" in log).

### 5. PLIF dead at start (this is OK)

**Symptom**: `spike_rate=0.0` for the first ~200 steps. Looks like
the spiking neurons aren't firing.

**Why**: This is by design. PLIFCell DA-LIF init uses a bimodal
threshold split — half the channels start with high threshold, fire
rarely, half with low, fire eagerly. The dense CfC path carries the
loss while PLIF channels train into a meaningful firing regime.

**Workaround**: none. Wait. spike_rate should rise to 0.05-0.30 by
step ~500. If it's still 0.0 at step 5000 the surrogate gradient is
broken — see `bench_surrogate.py` to verify the ATan pass-through.
