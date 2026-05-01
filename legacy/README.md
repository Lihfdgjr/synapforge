# legacy/

These are historical training scripts kept for reference.

**Canonical** is `train_100m_kd.py` (pretrain + KD) and `train_100m_sft.py`
(instruction-tune). See `docs/MASTER_PLAN.md` §3 for the phased recipe.

## Inventory

| File                          | Origin                          | Replaced by                          |
| ----------------------------- | ------------------------------- | ------------------------------------ |
| `train_100m.py`               | early dense-loss baseline       | `train_100m_kd.py` (phase 0/1)       |
| `train_3d.py`                 | 3D world-model probe            | (folded into `train_100m_kd.py` aux) |
| `train_full_modal.py`         | 9-modal byte-patch experiment   | (deferred; see §6 P10)               |
| `train_multimodal.py`         | early image+text trainer        | (folded into `train_100m_kd.py` aux) |
| `train_v15_full.py`           | v1.5 self-learn experiment      | `train_100m_kd.py`                   |
| `train_v16_unified.py`        | v1.6 unified backbone           | `train_100m_kd.py`                   |
| `train_v18_full_self.py`      | v1.8 self-learn variant         | `train_100m_kd.py`                   |
| `train_native_unified.py`     | byte-native modal pretrain      | `train_100m_kd.py`                   |
| `train_v42_universal.py`      | v4.2 universal trainer          | `train_100m_kd.py`                   |
| `synapforge_train.py`         | first package-internal trainer  | `train_100m_kd.py`                   |

## Running a legacy script

These are kept for historical reproducibility, not maintained. They may
import paths that have moved or use flags the current code no longer
supports. Use at your own risk:

```bash
python legacy/train_v42_universal.py --help   # may not work; for reference
```

For anything new, use the canonical pair at the repo root.
