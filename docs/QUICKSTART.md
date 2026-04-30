# Quickstart

## Local sanity test (no GPU, no remote)

```bash
git clone https://github.com/Lihfdgjr/synapforge
cd synapforge
git checkout v4.2-staging  # current dev branch

python -m pip install -r requirements.txt
python -c "from synapforge.action import HNSWSkillIndex, HierarchicalCodebook; print('OK')"
python -c "from synapforge.defense import PoisonDetector, ProvenanceTracker; print('OK')"
python -c "from synapforge.safety import RedBlueSelfPlay; print('OK')"
```

## v4.2 training on rental (A100×2 80GB)

```bash
ssh root@111.51.90.14 -p 44938

# On rental:
cd /workspace
git clone https://github.com/Lihfdgjr/synapforge synapforge_git
cp -r synapforge_git/synapforge /workspace/synapforge

# Pre-download datasets (~60GB)
python -m synapforge.data.fetch_datasets \
  --out /workspace/data \
  --include fineweb-en,alpaca-zh,alpaca-en,gsm8k,agent_math_gold

# Warmstart from previous best ckpt (v4.1)
python /workspace/synapforge/train_v42_universal.py \
  --warmstart /workspace/runs/synapforge_v41_neuromcp/best.pt \
  --out-dir /workspace/runs/synapforge_v42_universal \
  --teacher-path Qwen/Qwen2.5-0.5B \
  --steps 60000 --batch-size 8 --seq-len 1024 --grad-accum 2 \
  --lr 2e-4 --kd-weight 0.3 \
  --neuromcp-enabled \
  --skill-log /workspace/runs/skill_log.json
```

Expected:
- step 0: ce ≈ 4.0 (warmstart honored)
- step 100: ppl 40-80
- step 30k (5h): ppl 25-35, basic chat works
- step 60k (10h): ppl 20-28, multi-turn coherent

## Autonomous web learning daemon

Replaces hardcoded bilibili daemon. Picks high-FreeEnergy topics, multi-source
search, runs 7-gate WebPoisonGate.

```bash
# On rental:
nohup python -m synapforge.learn.autonomous_daemon \
  --out /workspace/data/web_cache.jsonl \
  --interval-min 30 \
  --topics-per-cycle 6 \
  > /workspace/runs/autolearn.log 2>&1 &
disown
```

Each cycle (every 30 min):
1. SelfGoalProposer picks 6 topics with lowest coverage in `web_cache.jsonl`
2. Multi-source search (bilibili / arxiv / wikipedia)
3. WebPoisonGate filters via 7 gates (see `docs/CONTINUAL_LEARNING.md`)
4. Survivors append to `web_cache.jsonl` with full provenance

Watch:
```bash
tail -f /workspace/runs/autolearn.log
wc -l /workspace/data/web_cache.jsonl
```

## Inference (chat) — after v4.2 finishes

```python
import torch
from synapforge.model_chat_600m import SynapForgeChat600M
from synapforge.action import HNSWSkillIndex
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = SynapForgeChat600M()
model.load_state_dict(torch.load("/workspace/runs/synapforge_v42_universal/best.pt")["model"])
model.eval()
model.cuda()

# Restore grown skills from previous sessions
skill_index = HNSWSkillIndex(dim=1024, index_dir="/workspace/runs/skill_index")

# Simple chat
prompt = "<|im_start|>user\n求方程 x^2 - 5x + 6 = 0 的解<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(prompt, return_tensors="pt").cuda()
with torch.no_grad():
    h = model.encode(ids)
    logits = model.lm_logits(h)
    next_tok = logits[0, -1].argmax()
    print(tok.decode(next_tok))
```

## Safety pipeline (post-base-training)

```bash
# Stage 0 — SFT refusal
python -m synapforge.safety.train_sft_refusal \
  --base-ckpt /workspace/runs/synapforge_v42_universal/best.pt \
  --hh-rlhf-path /workspace/data/hh-rlhf-refusal-5k.jsonl \
  --epochs 1

# Stage 1 — Constitutional AI critique-revise
python -m synapforge.safety.run_cai \
  --base-ckpt /workspace/runs/sft_refusal/best.pt \
  --red-prompts /workspace/data/red_prompts_2k.jsonl \
  --n-iters 4

# Stage 2 — Red-Blue DPO self-play (3000 pairs)
python -m synapforge.safety.run_dpo_selfplay \
  --base-ckpt /workspace/runs/cai/best.pt \
  --pairs 3000 --beta 0.1 --lora-rank 16 \
  --judge-mode hybrid  # rule + DeepSeek-V3 API
```

(Note: `safety.train_sft_refusal` / `safety.run_cai` / `safety.run_dpo_selfplay`
are entry points on the roadmap, not yet implemented.)

## Smoke test (no real training, no GPU)

```bash
cd /workspace
python -c "
import torch
from synapforge.action import (
    HNSWSkillIndex, HierarchicalCodebook, PerDomainNeuroMCP, SkillLog,
)
from synapforge.defense import (
    PoisonDetector, ProvenanceTracker, WebPoisonGate, ChatPoisonGate, WeightFirewall,
)
from synapforge.defense.gates import SourceBudget
from synapforge.safety import (
    RedBlueSelfPlay, ConstitutionalRevisor, AIJudge, DPOTrainer, sample_attack_prompt,
)
from synapforge.thinking import LatentThinker, CurriculumScheduler
from synapforge.moe import MoEFFN
from synapforge.learn import AutonomousLearnDaemon, RetrievalMemory

print('all 21 modules importable')

# Quick test of each
log = SkillLog('/tmp/_test_skill_log.json', save_every=1)
emb = torch.randn(1024)
pid = log.register('math', emb)
log.activate(pid, reward=0.8)
print(f'SkillLog: pid={pid} stats={log.stats()}')

idx = HNSWSkillIndex(dim=64, index_dir='/tmp/_test_hnsw')
for i in range(50):
    idx.add(domain='math' if i%2 else 'chat', embedding=torch.randn(64))
hits = idx.query(torch.randn(64), top_k=4)
print(f'HNSW: K={idx.K} top_k_hit_count={len(hits)}')

mcp = PerDomainNeuroMCP(hidden=64, action_dim=16)
h = torch.randn(2, 64)
action, ent, info = mcp(h)
print(f'PerDomainNeuroMCP: action={action.shape} ent={ent:.3f}')

cat, prompt, sev = sample_attack_prompt()
print(f'Red attack: cat={cat} sev={sev} prompt={prompt[:40]}')
print('SMOKE OK')
"
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ImportError: synapforge.learn` | Workdir not /workspace | `cd /workspace && python -m ...` |
| `Path("Qwen/...").exists() == False` (kd=0) | HF repo id rejected by Path check | Trainer patched in v4.2-staging — pull latest |
| OOM at z-loss step | bs × seq × vocab > 8GB | Drop `--batch-size` to 4 or `--seq-len` to 1024 |
| ppl=1.4 too low | response-only mask hides 95% tokens | Add `--log-mask-ratio --log-ce-full` (planned flag) |
| tok/s 1100 (17× slow) | PerDomainNeuroMCP runs all 4 heads | Use `--neuromcp-routing top1` (planned flag) |
| SSH MaxStartups exhausted | Too many failed reconnects | Wait 10-25 min, sshd backlog clears |
