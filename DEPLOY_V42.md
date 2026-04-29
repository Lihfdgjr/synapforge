# v4.2 Deployment

Once rental SSH is reachable (111.51.90.14:44938), upload the staging tree:

```bash
# from D:/ai_tool/synapforge_v42_staging:
rsync -avz --progress \
  synapforge/action/ root@111.51.90.14:/workspace/synapforge/action/ \
  -e "ssh -p 44938"

rsync -avz --progress \
  synapforge/thinking/ root@111.51.90.14:/workspace/synapforge/thinking/ \
  -e "ssh -p 44938"

rsync -avz --progress \
  synapforge/moe/ root@111.51.90.14:/workspace/synapforge/moe/ \
  -e "ssh -p 44938"

scp -P 44938 \
  synapforge/train_v42_universal.py \
  root@111.51.90.14:/workspace/synapforge/train_v42_universal.py
```

After upload, on the rental:

```bash
cd /workspace
# 1. Verify imports
python -c "from synapforge.action import PerDomainNeuroMCP, SkillLog; \
           from synapforge.thinking import LatentThinker, CurriculumScheduler; \
           from synapforge.moe import MoEFFN; print('OK')"

# 2. Start v4.2 training (warmstart from v4.1 best)
python -m synapforge.train_v42_universal \
  --warmstart /workspace/runs/synapforge_v41_neuromcp/best.pt \
  --out-dir /workspace/runs/synapforge_v42_universal \
  --steps 60000 --batch-size 8 --seq-len 2048 \
  --lr 2e-4 --kd-weight 0.3 --novelty-weight 0.05 \
  --neuromcp-enabled --coconut-enabled \
  > /workspace/runs/synapforge_v42_universal/train.log 2>&1 &

# 3. Monitor (in another shell)
tail -f /workspace/runs/synapforge_v42_universal/train.log

# 4. Inspect skill log persistence
cat /workspace/runs/skill_log.json | python -m json.tool | head -30
```

## Components (1441 lines)

| File                                    | Lines | Role                                                   |
|-----------------------------------------|-------|--------------------------------------------------------|
| action/skill_log.py                     | 216   | JSON-backed skill memory (LTP/LTD, query_similar)      |
| action/per_domain_neuromcp.py           | 387   | 4-domain heads + router + on-demand skill spawn        |
| thinking/coconut.py                     | 168   | LatentThinker + curriculum + pause-token injector       |
| moe/expert_ffn.py                       | 138   | MoE FFN (8 routed top-2 + 1 shared, noisy gate)        |
| train_v42_universal.py                  | 503   | Trainer: CE+KD+entropy+lb+STDP loss, neuromodulator    |

## Training delta vs v4.1

- Coconut `<bot>/<eot>/<pause>` tokens added to Qwen tokenizer (3 new ids)
- Per-domain NeuroMCP replaces global NeuroMCPHead — 4 codebooks (math/chat/code/web)
- skill_log.json restored on boot, frozen prototypes for previously-grown skills
- Three-factor STDP aux loss with M_t = 0.5·(1/(1+FE)) + 0.3·novelty - 0.2·homeo_drift
- MoE FFN swap is `--moe-enabled` flag (default off, save first dense run for compare)
