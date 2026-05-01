# 不用 LoRA 的战略原因 — Anti-LoRA Position

**铁律 2026-05-01**: SynapForge / Synap-1 demo 路径 + 论文 baseline **零 LoRA / 零 transformer fallback**.

## 为什么这条线绝对不能越

我们的卖点是"对现有架构（transformer）的突破"——LNN+SNN 是另一条路。LoRA 是 transformer 上的 patch；用 LoRA 当 demo 的"insurance"路径会同时摧毁两个东西：

1. **架构故事破产**。pitch 说"跳出 transformer"，演示却用 LoRA-on-Qwen 兜底——投资人合理质疑：如果你自己都用 transformer 兜底，为什么我应该相信你 LNN+SNN 这条路能成？
2. **论文不可投**。任何顶会（NeurIPS/ICLR/ICML）的 reviewer 看到 "Synap-1 chat ability is enabled by Qwen+LoRA fallback" 立刻拒：你测的是 Qwen 不是你的架构。reproducibility 与 architecture-claim severability 都崩。
3. **科学诚信**。"5 个差异化能力"全建立在 LNN+SNN 上（NeuroMCP 突触生长、R-fold 闭式 CfC、inference-STDP 单 LOC unlock、LNN+SNN backbone、triple-backup）。LoRA 路径里这些**一个都不用**——是承认主线 stack 不可用。

## 已删除的 Plan C 痕迹（commit 2026-05-01）

```
git rm scripts/launch_plan_c_cpu.sh
git rm scripts/train_qwen_lora.py
git rm scripts/qwen_lora_chat_repl.py
git rm docs/PLAN_C_CPU_NOTES.md
git rm docs/PLAN_C_QWEN_LORA.md
git rm docs/PLAN_C_RUNBOOK.md
git rm synapforge/demo/qwen_lora_demo.py
edit  synapforge/demo/cli.py     # remove cmd_qwenchat + qwenchat subparser
edit  scripts/chat_eval_gate.py  # remove _is_plan_c_ckpt + Qwen-LoRA route
edit  docs/MASTER_PLAN.md        # remove O7 Plan C objective + P5 entry
edit  docs/INVESTOR.md           # remove Plan C disclosure block
edit  docs/PROGRESS.md           # remove Plan C row in backup matrix
```

## 新的 insurance 路径（纯 Synap-1 native）

如果 Synap-1 训练遇到极端故障，**唯一可接受的兜底**是：

### Option A: 缩小 Synap （30M / 50M）跑短训
- 减少参数量到 30M-50M LNN+SNN，跑 6-12h 而不是 24h+
- 更可能在投资人 demo 时间窗内拿到 chat-grade ckpt
- **依然是 LNN+SNN 架构** — 故事 + 论文都成立

### Option B: 录播 demo（chat_recorded.json）
- v4.x 时期已经做过：`synapforge/demo/chat_recorded.json` 存 5 EN + 5 ZH 真实输出
- 现场 live 失败时回放历史 healthy ckpt 的输出
- **诚实声明**: "这是 Synap-1 v4.1 的录播，本次 Run 3c 还在收敛"
- 投资人接受度 > 用别人的模型兜底

### Option C: 演示训练曲线 + 架构而不是 chat
- 把 demo 重心从 "live chat" 移到 "5 个差异化能力 + 训练曲线 + 架构图"
- NeuroMCP / R-fold / STDP / multimodal byte-patch 全是**机制层面**演示，不靠 chat ckpt
- **chat 是锦上添花，不是核心**

## 论文上的等价铁律

baseline 表格里：
- ✅ 可比对 Mamba / RetNet / Hyena / RWKV（**非-attention** 架构 baseline，公平对比）
- ✅ 可比对 same-param 范围 transformer (SmolLM2-360M / TinyLlama-1.1B / Qwen2.5-0.5B)，但仅作为 reference upper bound
- ❌ **绝不**用 LoRA-on-X 作为我们 stack 的扩展或 fallback
- ❌ **绝不**用 Qwen-as-base 让 Synap 学会某能力——KD distillation **OK**（teacher 是黑盒，logits 只是软标签，student 仍 100% LNN+SNN），LoRA-on-Qwen **NOT OK**（产出不是 LNN+SNN）

## KD 仍然 OK 的边界条件

我们当前用 Qwen 0.5B 作为 KD teacher（`--teacher Qwen/Qwen2.5-0.5B`）。这是**合法**的：

- Teacher 是 frozen 黑盒，只输出 logits
- Student 100% 是 SynapForge100M（LNN+SNN），KD 的 KL loss 只把 teacher 软标签当 supervision signal
- 论文里 KD baseline 是行业惯例（DistilBERT / TinyBERT / SmolLM 都这样做）
- 投出去的 ckpt **完全不依赖** teacher 在推理时存在
- 公平对比：跟 Mamba 论文等 "用 transformer KD 训 SSM" 的范式一致

**LoRA 不一样**：LoRA 在推理时**必须**和 base model 协同工作。删 base 就死。

## 行动

- 任何 future agent 想加 "LoRA fallback" 或 "transformer fallback frontend" → 拒绝。
- 任何 demo / pitch / paper figure 不出现 "Plan C" / "Qwen-LoRA" 字样。
- continual learning 内部如果用 "low-rank update on CfC weights" 这种术语没问题——那是 LNN 自己的技巧，**不是** LoRA-on-transformer。

— 2026-05-01
