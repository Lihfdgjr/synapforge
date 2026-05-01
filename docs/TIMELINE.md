# Synap-1 Timeline — 真实 ETA

**最后更新**: 2026-05-02 01:51（Run 3m step 6770, VAL 7004 holdout）

## ⚠️ 重大修正: lm-head-spectral-norm 重置成本

Run 3l 灾难发散后 (step5500 val=2522), Run 3m 用 step_002000.pt warmstart + 启 T2.6
`--lm-head-spectral-norm`. 该参数包装 lm_head 为 `weight_orig` + `weight_v`/`weight_u`,
但 step_002000.pt 只存了 `weight` — 加载时 spectral_norm 模块用未训练的 v/u 重新归一化,
**有效让 lm_head 完全重置**. 损失代价: step 1 ce=11.93 (随机初始化水平).

恢复轨迹（实测）:
| step | VAL ppl holdout | Δ vs prev | Δ rate |
|------|----------------|-----------|--------|
| 3000 | 16031 | — | — |
| 4000 | 11838 | -4193 | -26% |
| 5500 | 8226  | -3612 | -31% |
| 6000 | 7460  | -766  | -9%  |
| 6500 | 7004  | -456  | -6%  |
| 6770 (train ce 8.99) | — | — | — |

**关键发现**: 恢复**减速** (logarithmic). step 4000→5500 -1500步降 -3612 ppl,
step 6000→6500 -500步只降 -456 ppl. 后期斜率 ~-1 ppl/step.

**新 ETA 推算** (基于 -1 ppl/step 后期斜率):
- val ≤ 6000 (next milestone, no flag change): step ~7770, **+10 min** (02:01)
- val ≤ 1000: step ~12500, +1.5h (03:30)
- val ≤ 500: step ~13000, +1.7h (03:40)
- val ≤ 250 (**phase 1 trigger**): step ~13500, **+2h (04:00 May 2)**
- val ≤ 100 (phase 2): step ~14500, +3h (05:00)
- val ≤ 60 (phase 3 trigger): step ~16000, +5h (06:50)

**vs 原计划 (Run 3l/3b 健康轨迹)**: 慢了 5-8h 因 lm-head 重置. 但 spectral-norm 防 z-loss 漂移
对长 horizon 有价值, 不全是 sunk cost.

---

> 真实数据驱动的时间表。不是 aspirational schedule，是基于已发生 GPU-h 实测推算的剩余时间。每次 turn 都更新这份文档。

---

## 当前坐标（Run 3e）

| 度量 | 数值 |
|------|------|
| Trainer | PID 19790 (Run 3e) live on rental 117.74.66.77 |
| Warmstart | `step_002000.pt` 含 optim_state（"接着训练，不重训"路线）|
| Last VAL | step 1000 ppl=**355** (holdout) |
| Last train ce | step 1100 = 6.49 (KD on) |
| Throughput | 18.8k tok/s @ bs=64 seq=256 (≈ 0.83s/step) |
| GPU util | 74% / 77GB / 80GB |

---

## 已花费 GPU-h（5 次 run 的真实账）

| Run | 持续时间 | 终止原因 | 学到的事 |
|-----|---------|---------|---------|
| Run 1 (April) | 4h | word salad 发散 | PLIF 阈值 + cell 类型错 |
| Run 2 (May 1 13:58-18:01) | 4h | 磁盘 99.9% 满 → torch.save 崩 | SAVE_EVERY=250 太频，必须 ckpt cleanup |
| Run 3a | 30 min | Adam stale momentum 中毒 | strip optim_state on warmstart |
| Run 3b | 1h | LR=3e-4 太大 + data 顺序 → step 5500 ppl 4071 | LR=1e-4 |
| Run 3c | 1.5h | step 2500 同位置 ppl 1886 | data 必须 shuffle |
| Run 3d (cold optim) | 5 min | 用户要求"接着训练" → 切到 3e | 不要 cold-start |
| **Run 3e** | running 30 min | 还在训 | 看 §下文 |

**累计 GPU 时间**: ~12 小时（含 4 次失败 run 的 debug 时间）。
**有效进度**: Synap-1 backbone 在 Run 3c step 2000 ckpt 上，VAL 397 → Run 3e 起步 320，缓慢但单调下降。

---

## 剩余 ETA（按实测速度推算）

每行包含：触发条件 / 估计步数 / 估计 GPU-h / 实际 wall-clock。

### Phase 0 → 1 触发（val ppl ≤ 250）
- **当前**: VAL 320 (Run 3e step 1000)
- **需要降**: 320 → 250 = ce 5.77 → 5.52 = 0.25 ce
- **每 1000 步 ce 降**: ~0.05 (实测 Run 3c 同区间)
- **估计步数**: 4000-6000 步
- **GPU 时间**: 4000 × 0.83s = **55 分钟** 至 6000 × 0.83s = **83 分钟**
- **Wall clock ETA**: 22:08 + 1-1.5h = **23:08-23:30 今晚**
- **风险**: 如果 step 2500 仍发散（shuffle 没解决），需回到 debug，+12h

### Phase 1 → 2 触发（val ppl ≤ 100）
- 加入 TTT replay + curiosity 0.05（自动 phase relauncher 生效）
- **估计步数**: 5000-10000 步（curiosity ramp 0→0.05 用 1500 步，然后纯训）
- **GPU 时间**: ~70-140 分钟 = **1.5-2.5 小时**
- **Wall clock**: 完成 phase 1 大约**明天凌晨 02:00-04:00**

### Phase 2 → 3 触发（val ppl ≤ 60）
- 加入 multimodal aux（image+audio+time-series byte-patch）
- **风险**: Run 1 burned 2× by enabling cross-modal aux too early
- **估计步数**: 4000-8000 步
- **GPU 时间**: **1-2 小时**
- **Wall clock**: **明天上午 04:00-06:00**

### Phase 3 SFT (alpaca-zh response-only)
- 数据已就绪: `/workspace/data/alpaca_zh_qwen_tokenized.parquet` (48,818 examples)
- **估计步数**: 2000-4000 步 (lr=1e-4)
- **GPU 时间**: **30 分钟 - 1 小时**
- **Wall clock**: **明天上午 05:00-07:00**

### Phase 4 RL (GRPO + sympy verifier)
- 数学验证 RL，gsm8k subset
- **估计步数**: 5000-10000 RL rollout
- **GPU 时间**: **1.5-2.5 小时**
- **Wall clock**: **明天上午 07:00-10:00**

---

## 投资人 demo ready 时间表

| 阶段 | Wall-clock ETA | Status |
|------|---------------|--------|
| ce ≤ 5.5 (val ppl ≤ 250 ≈ phase 1) | **今晚 23:00** | optimistic |
| ce ≤ 4.6 (val ppl ≤ 100 ≈ phase 2) | **明早 02-04 点** | best case |
| ce ≤ 4.1 (val ppl ≤ 60 ≈ phase 3 chat-grade) | **明早 04-06 点** | best case |
| chat eval ≥ 60% pass | **明早 07-10 点** | best case |
| 投资人可看 live demo | **明天下午 14:00-18:00** | best case + buffer for issues |

**Honest pad**: 加 50% buffer 防意外（rental 死、又一次发散、debug）→ **48h within target ready**。

---

## 高 ROI 的下一步（提速选项）

如果想提前 ready，**下次 trainer restart 时启用**（perf v2 commit `5a3ecef` 已 ready）：

1. `--prefetch-factor 4 --pin-memory` → +10-20% throughput
2. `--kd-every-adaptive` → +20-30% averaged（KD 频率自适应：student 远 → kd-every=2，近 → kd-every=16）
3. bs 64 → 80 (perf 已 commit `4d0d2a9`，需 sparse z-loss top-K=2048 一起开) → +25%
4. `--torch-compile reduce-overhead` → +5-15%（实验性，可能 NaN 回滚）

**复合预期**: 18.8k → **35-45k tok/s** = **加速 2×**，phase 1 trigger 提前到 **23:00 → 22:30**, phase 3 提前到 **明早 04:00 → 03:00**。

但**不要打断当前 Run 3e**（接着训练原则）。下次 trainer 自然死亡或 phase 1 自动触发时，phase auto-relauncher (commit `9bcf49f`) 会把这些 perf flags 一起带上。

---

## 风险登记（按发生概率排序）

| 风险 | 概率 | 影响 | Mitigation |
|------|------|------|-----------|
| Run 3e 在 step 2500 又发散 | 15% | +12h debug | shuffle ON 已在该 ckpt 起作用；如真发散看 Master Plan §6 P25（待写）|
| Rental SSH 飘断 → trainer 失控 | 30% | +30min 重连 | systemd-run watchdog (P8) 已 commit，未启用；下次重启时启 |
| PLIF 永远 dead | 20% | phase 3 chat eval 卡 60% | P1 homeostasis 在跑（thr ↓ 0.0500→0.0495 step 50 已观察）|
| 磁盘再满 | 10% | +1h 清理 | SAVE_EVERY=2000 + ckpt_cleanup background loop |
| 投资人时间提前 | – | demo 不够熟 | INSURANCE_NATIVE.md 三选项（Synap-Mini / 录播 / mechanism-pivot）|

---

## 这份文档怎么更新

每次 turn 末尾或 milestone 触发时：
1. 更新顶部"最后更新" + "当前坐标" 表
2. 在 §"已花费" 加新 Run 行
3. §"剩余 ETA" 调整步数估计如果实测速度变了
4. §"投资人 ready" 时间根据 ETA 重算

下次 turn 必读：`PROGRESS.md`（实时 metrics）+ `MASTER_PLAN.md` §6/§7 + 这份 `TIMELINE.md`。
