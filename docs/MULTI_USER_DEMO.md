# Multi-User Chat Demo (verbatim transcript)

- Storage root: `C:\Users\26979\AppData\Local\Temp\synapforge_demo_mem_qqkf9bfg`
- VRAM delta: **0 bytes** (zero is the contract)
- CUDA present: no (CPU/disk-only by design)

## Turn-by-turn

- **alice / user**: 我喜欢猫
- **alice / assistant**: 好的,我记下了你喜欢猫。
- **bob / user**: 我讨厌猫
- **bob / assistant**: 好的,我记下了你讨厌猫。
- **alice / user**: 你记得我喜欢什么吗
- **alice / assistant**: 你喜欢猫。
- **bob / user**: 你记得我喜欢什么吗
- **bob / assistant**: 你讨厌猫。

## Cross-user leak check

- alice top-1 recall: `我喜欢猫`
- bob   top-1 recall: `我讨厌猫`

Neither user's hit list contains the other's prior message — namespace
isolation enforced at the API surface AND at the filesystem layer.

## What you can poke at on disk

```
C:\Users\26979\AppData\Local\Temp\synapforge_demo_mem_qqkf9bfg/
├── alice/
│   ├── log.jsonl       # alice's conversation history
│   ├── prefs.json      # {{likes_cats: true}}
│   └── index.bin       # reserved (HNSW/FAISS drop-in)
└── bob/
    ├── log.jsonl
    ├── prefs.json      # {{hates_cats: true}}
    └── index.bin
```

Run: `python scripts/demo_multi_user_chat.py`
