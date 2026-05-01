# ARCHIVE: arxiv scans (cs.NE / cs.LG)

Auto-appended by cron task T2.1. Each scan filters last 7 days of arxiv listings on
keywords: ternary, matmul-free, spiking, liquid, STDP, SNN, LIF, PLIF, surrogate,
neuromorphic, BitNet, EnergyToken, continual-learning. Keep 1-paragraph synth per
paper to support Synap-1 architecture decisions.

---

## 2026-05-01 — A Multiplication-Free Spike-Time Learning Algorithm and its Efficient FPGA Implementation for On-Chip SNN Training
arxiv:2604.23218  authors: Maryam Mirsadeghi, Mojtaba Mirbagheri, Saeed Reza Kheradpisheh
**Claim**: Spike-time supervised training with zero floating-point multiplies and no explicit gradient storage; fully event-driven digital pipeline reaches 96.5% MNIST / 84.8% Fashion-MNIST on FPGA.
**Applicability to Synap-1**: Direct fit for our PLIF spike path — replaces surrogate-gradient BPTT with mul-free local rule, matching our triton_block backend's matmul-free direction (BitNet b1.58 task #117). Useful as a STDP-adjacent baseline for the inference-time STDP unlock work; could let us drop fp32 multiplies in the spike branch entirely on CPU/edge deployments.

## 2026-04-30 — NORACL: Neurogenesis for Oracle-free Resource-Adaptive Continual Learning
arxiv:2604.27031  authors: Karthik Charan Raghunathan, Christian Metzner, Laura Kriener, Melika Payvand
**Claim**: Resolves stability-plasticity tension by growing the network when two saturation signals (representational + plastic) trip — no oracle task boundaries; dissimilar tasks expand feature-extraction layers, similar tasks expand later combination layers.
**Applicability to Synap-1**: Strong alignment with our SparseSynapticLayer + DynamicCodebook (NeuroMCP, task #136 / #223) which already grows synapses + prototypes. NORACL's two-signal saturation trigger is more principled than our current heuristic codebook growth; worth porting the saturation monitors as a gating signal for K-prototype expansion in v4.2 continual-learning track.

## 2026-04-30 — Learning to Forget: Continual Learning with Adaptive Weight Decay
arxiv:2604.27063  authors: Aditya A. Ramesh, Alex Lewandowski, Jürgen Schmidhuber
**Claim**: FADE — per-parameter weight-decay rates adapted via approximate meta-gradient descent; stable-knowledge params decay slowly, drifting params decay fast.
**Applicability to Synap-1**: Plugs into Track A (web continual training, EWC + LoRA shadow merge) from the continual-vs-poison balance plan. Per-parameter decay is a cheap upgrade to uniform L2 in train_100m_kd.py and complements our existing TRAK gating. Not bio-inspired so does not touch CfC/PLIF, but fits the optimizer side of the v4.2 self-learn loop.

## 2026-05-01 — EdgeSpike: Spiking Neural Networks for Low-Power Autonomous Sensing in Edge IoT Architectures
arxiv:2604.27004  authors: Gustav Olaf Yunus Laitinen-Fredriksson Lundstrom-Imanov, Taner Yilmaz
**Claim**: End-to-end SNN co-design — hybrid surrogate-gradient + direct-encoding training, hardware-aware NAS, multi-target runtime (Loihi 2 / SpiNNaker 2 / Cortex-M); 91.4% mean acc with 18-47x energy reduction on neuromorphic, 4.6-7.9x on Cortex-M; 7-month 64-node deployment shows 6.3x battery extension with on-device local-plasticity continual learning bounding drift to 0.7pp.
**Applicability to Synap-1**: Strongest validation yet of our local-plasticity continual-learning claim (drift 0.7pp vs 2.1pp without). EdgeSpike's spike-sparse SIMD kernels for Cortex-M parallel our cpu_avx2 backend goals; lift their 12-point energy/accuracy Pareto-front methodology for our v1.0 hardware investor pitch. Hybrid surrogate+direct-encoding training is a candidate for the v1.5 dense-CfC bypass schedule.

## 2026-05-01 — NeuroRing: Scaling Spiking Neural Networks via Multi-FPGA Bidirectional Ring Topologies and Stream-Dataflow Architectures
arxiv:2604.28059  authors: Muhammad Ihsan Al Hafiz, Artur Podobas
**Claim**: HLS-on-FPGA SNN accelerator with bidirectional ring + stream-dataflow architecture; real-time factor 0.83 on full-scale cortical microcircuit benchmark across two FPGAs with strong + weak scaling.
**Applicability to Synap-1**: Inference-only and FPGA-specific — limited direct port to our PyTorch + triton_block training stack. Useful as long-horizon reference for Synap-1 hardware story (post-v1.0 neuromorphic deployment); the stream-dataflow + ring topology could inform how our heterogeneous GPU+CPU cluster (task #157) shards spike traffic between nodes.
