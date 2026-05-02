# Decorative -> Real Feature Audit

Generated: 2026-05-02 23:48:47
Log probe: (not run -- pass --log to enable)

## Summary

| Feature | file_exists | tag/flag set | flag in launcher | actually_active | notes |
| --- | --- | --- | --- | --- | --- |
| STDP-only routing on SparseSynapticLayer | PASS | PASS | PASS | (no log) |  |
| Multimodal byte-patch (image/audio/time_series) | PASS | PASS | PASS | (no log) | phase-aware gates this until val_ppl <= 100 (Phase 2) |
| Web daemon -> trainer parquet pipe | PASS | PASS | PASS | (no log) | default weight 0.10 in --data-files; daemon must be running for any rows to land |
| R-fold parallel scan in TRAINING (not just inference) | PASS | PASS | PASS | (no log) | backward path verified by tests/cells/test_rfold_equivalence.py + tests/cells/test_rfold_train_bit_exact.py |
| Ternary BitNet QAT (delta_proj/b_proj) | PASS | PASS | FAIL | BLOCKED | DEFERRED -- ternary swap on warmstart triggers LM-head reset (feedback_spectral_norm_warmstart_cost.md). Needs fresh run, not warmstart. |
| PLIF spike revival (Run 7 dense bypass + sparse spike) | PASS | PASS | PASS | BLOCKED | DEPENDENCY -- spike density only emerges post step 4000 (--plif-dense-bypass-steps 4000). Verify in train log AFTER step 4000+. |

## Detail per feature

### STDP-only routing on SparseSynapticLayer

* file: `D:\tmp\sf_activate_decorative\synapforge\action\neuromcp.py`
* tag/flag: `self.weight._sf_grad_source = ['stdp']`
* file_exists: **PASS**
* tag/flag set in code: **PASS**
* flag wired in Run 8 full launcher: **PASS**
* actually-active in log: **(no log)**

### Multimodal byte-patch (image/audio/time_series)

* file: `D:\tmp\sf_activate_decorative\synapforge\trainer_mixins.py`
* tag/flag: `--modal-list / --modal-data-dir / --modal-alpha`
* file_exists: **PASS**
* tag/flag set in code: **PASS**
* flag wired in Run 8 full launcher: **PASS**
* actually-active in log: **(no log)**
* notes:
    - phase-aware gates this until val_ppl <= 100 (Phase 2)

### Web daemon -> trainer parquet pipe

* file: `D:\tmp\sf_activate_decorative\synapforge\data\web_daemon_sink.py`
* tag/flag: `WebDaemonSink + scripts/web_self_learn_daemon.py`
* file_exists: **PASS**
* tag/flag set in code: **PASS**
* flag wired in Run 8 full launcher: **PASS**
* actually-active in log: **(no log)**
* notes:
    - default weight 0.10 in --data-files; daemon must be running for any rows to land

### R-fold parallel scan in TRAINING (not just inference)

* file: `D:\tmp\sf_activate_decorative\synapforge\cells\liquid.py`
* tag/flag: `--rfold + --rfold-chunk`
* file_exists: **PASS**
* tag/flag set in code: **PASS**
* flag wired in Run 8 full launcher: **PASS**
* actually-active in log: **(no log)**
* notes:
    - backward path verified by tests/cells/test_rfold_equivalence.py + tests/cells/test_rfold_train_bit_exact.py

### Ternary BitNet QAT (delta_proj/b_proj)

* file: `D:\tmp\sf_activate_decorative\synapforge\cells\liquid.py`
* tag/flag: `--weight-quant ternary`
* file_exists: **PASS**
* tag/flag set in code: **PASS**
* flag wired in Run 8 full launcher: **FAIL**
* actually-active in log: **(no log)**
* BLOCKED: DEFERRED -- ternary swap on warmstart triggers LM-head reset (feedback_spectral_norm_warmstart_cost.md). Needs fresh run, not warmstart.

### PLIF spike revival (Run 7 dense bypass + sparse spike)

* file: `D:\tmp\sf_activate_decorative\synapforge\cells\plif.py`
* tag/flag: `--plif-dense-bypass-steps + --sparse-spike-synapse`
* file_exists: **PASS**
* tag/flag set in code: **PASS**
* flag wired in Run 8 full launcher: **PASS**
* actually-active in log: **(no log)**
* BLOCKED: DEPENDENCY -- spike density only emerges post step 4000 (--plif-dense-bypass-steps 4000). Verify in train log AFTER step 4000+.
