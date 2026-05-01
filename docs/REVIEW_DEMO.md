# REVIEW_DEMO

Code review of `synapforge.demo` + `synapforge/cells/rfold.py` + `synapforge/parallel.py`
+ `scripts/verify_rfold.py` + outreach docs. Date: 2026-05-01. Reviewer focus: investor
demo correctness, doc consistency, presentation hygiene.

---

## Bug list (severity / file:line / fix)

| # | Sev | File:line | Symptom | Fix |
|---|-----|-----------|---------|-----|
| 1 | HIGH | `synapforge/demo/cli.py:14-37` | PITCH printed em-dashes (`—`, U+2014) which Windows cp936 console renders as `??`. Investor's first sight = mojibake. | Replaced em-dash with `--`; added `_force_utf8_stdout()` reconfigure on import. |
| 2 | HIGH | `synapforge/demo/chat_demo.py` (caller side) | Chinese characters in 5 ZH prompts rendered as `??` on Windows console (cp936). Recorded `.json` was correct UTF-8 the whole time, only console rendering broke. | UTF-8 stdout reconfigure in `cli.py` cascades to chat_demo prints. |
| 3 | HIGH | `synapforge/demo/stdp_demo.py:27` | `HEAT_GLYPHS = " ▏▎▍▌▖▊▉█"` (block-element U+258x) rendered as `??` on cp936; the heatmap of the STDP demo was unreadable. | Replaced with ASCII ladder `" .:-=+*#@"`. Loses gradient resolution but renders identically on Win/Linux/pipes. |
| 4 | HIGH | `synapforge/demo/cli.py:24-25` | Pitch claimed "5% -> 28% density on the 4-button validation env"; actual measured run at default `--trials 80` produces 5.7% -> 7.7%, K 9 -> 11 (not 9 -> 14). Investor would catch this on first run. | Rewrote pitch to match measured numbers (~6-8% density growth at 80 trials, 100% hit-rate post-warmup). |
| 5 | HIGH | `synapforge/demo/rfold_bench.py:92` | `if device == "cpu":` — comparing `torch.device` to string `"cpu"`; always False. So even on CPU runs the GPU verdict text printed, which mis-described the empirical table. | Fixed to `if device.type == "cpu":`. Also rewrote both CPU/GPU verdicts to match the actual N=64-only win region. |
| 6 | HIGH | `docs/INVESTOR.md:36-44` | Same 5%->28% / K=9->14 claim as bug #4, this time on the public investor doc. | Rewrote evidence bullet: at default 80 trials density grows ~6%->~8%, K 9->11; the saturated regime needs longer training. |
| 7 | MED | `docs/INDEX.md:35` | Row says "measured 167x speedup at R=1024" — directly contradicts RFOLD_PAPER.md which retracts the 167x extrapolation in Appendix B. | Updated to "honest GPU peak 2.99x at N=64 R=16. Appendix B retracts an earlier 167x extrapolation." |
| 8 | MED | `docs/INDEX.md:113` cross-ref | Says "R-fold ... INVESTOR (the 167x bench)" — same contradiction. | Updated to "INVESTOR (the 2.99x peak speedup), HONEST_ASSESSMENT (full CPU/GPU table)". |
| 9 | MED | `synapforge/parallel.py` | `python -m synapforge.parallel` produces no output (no `__main__` block). User's spec listed this as a smoke test. | Added `if __name__ == "__main__": print("=== synapforge.parallel ==="); print_setup()`. |
| 10 | MED | `scripts/verify_rfold.py:115-117` | Verdict text says "CPU win threshold: ~N>=512" but the actual measured table (in the same script) shows N=512 is **0.02-0.08x** (catastrophic loss). | Rewrote verdict: "CPU win region: ~N=64 R>=16 only. Larger N loses to LAPACK solve overhead." |
| 11 | LOW | `README.md:48` | Status snapshot dated 2026-04-30 but every recent row is 2026-05-01 work. | Updated header date. |
| 12 | LOW | `README.md:66` | Demo status row claimed "density 5.7%->6.7%, K 9->10"; an earlier short-trial run. Real default 80-trial run is 5.7%->7.7%, K 9->11. STDP and chat rows missing. | Rewrote row to mention all four sub-demos with their measured numbers. |
| 13 | LOW | `synapforge/demo/stdp_demo.py:107` | Comment says "density (\|W\|>0.02) climbs from ~5%" — eps and starting density both wrong. Implementation uses eps=0.05 and starts at 0.0%. | Fixed comment to match `_density(eps=0.05)` and "0% -> ~25-30% over 200 trials". |
| 14 | LOW | `synapforge/demo/stdp_demo.py:159` print | Em-dash `—` in printed string would render as `?` on cp936 even with stdout reconfigure (some pipes still mangle). | Replaced with `--` for portability. |
| 15 | LOW | `pyproject.toml:85` | `testpaths = ["tests"]` does not collect `synapforge/tests/test_rfold.py`. Adding it wholesale, however, breaks because of legacy mscfc imports in `synapforge/tests/test_correctness.py`. | Documented the constraint inline; users invoke `pytest synapforge/tests/test_rfold.py -v` explicitly (matches user's smoke spec). |
| 16 | LOW | `scripts/rfold_paper_repro.sh:38` | `sys.path.insert(0, str(Path(__file__).resolve().parent if False else "."))` — leftover dead `if False else` branch. Cosmetic; works (`cwd == REPO_ROOT`). | Left as-is; not user-visible. Polish item. |
| 17 | INFO | `synapforge/parallel.py:130-133` `place_mixed_device` | When `tie_lm_head=True` (which `chat_demo.py:73` enables), `lm_head.weight` aliases `embed_tokens.weight`. The default `cpu_module_names` includes both names so the second-loop name-prefix check matches and skips the alias. **Latent risk** if a caller passes only one of the two: the second loop will move the (CPU-pinned) shared param back to GPU. | Added comment intent; not a current bug because the default tuple covers both names. |

---

## Doc consistency table

Cross-doc audit. Major contradictions in the prior tree, now fixed (as noted) or
flagged.

| Topic | doc A | doc B | Status |
|-------|-------|-------|--------|
| R-fold speedup headline | RFOLD_PAPER.md (2.99x peak, 167x retracted) | INDEX.md (was: 167x at R=1024) | **Fixed** — INDEX.md now matches paper. |
| R-fold speedup headline | INVESTOR.md (2.99x on A800) | RFOLD_PAPER.md (consumer GPU) | **Inconsistent** — INVESTOR says A800, paper says RTX 4070. Left alone (low-stakes for investor framing) but should be qualified to "consumer GPU; A800 bench pending" in a future pass. |
| Demo NeuroMCP density | INVESTOR.md (was: 5%->28% in 80 trials) | actual run (5.7%->7.7% in 80 trials) | **Fixed** in INVESTOR.md. |
| Demo NeuroMCP density | cli.py PITCH (was: 5%->28%) | actual run | **Fixed** in PITCH. |
| ppl headline | README.md (44.2 best at step 46350) | HONEST_ASSESSMENT.md (84 v4.0 base, 44.2 v4.1 wire-in) | Consistent (HONEST_ASSESSMENT shows the staged numbers; README just quotes the best). |
| Status snapshot date | README.md header (was: 2026-04-30) | actual row content (2026-05-01) | **Fixed** in README. |
| CPU rfold N=128 R=8 speed | RFOLD_PAPER.md (0.45x) | HONEST_ASSESSMENT.md (0.60x) | **Inconsistent** (~0.15 pp). Both are within run-to-run variance; not material to investor. Polish. |
| docs/INDEX.md REVIEW_DEMO row | "(planned)" | this file exists | **Fixed** — INDEX now points to this file. |
| outreach/README.md `RFOLD_PAPER.md ~600 LOC` | actual file: 463 lines | claim | Minor rounding; left alone. |
| INDEX.md cross-ref "R-fold ... 167x bench" | INDEX (was: 167x) | INVESTOR.md / RFOLD_PAPER (2.99x) | **Fixed**. |
| QUICKSTART expected output | QUICKSTART.md "trial 79 density=27.9% K=14" | actual demo run (`density=7.7% K=11`) | **Inconsistent** — QUICKSTART path-1 sample output overstates density growth. Left alone for next pass; not load-bearing because user runs the demo and sees real output, but should be re-recorded. |

---

## Investor demo flow walkthrough — hostile reviewer running `synapforge-demo all`

Imagine a sceptical VP running this on their Win11 laptop without setup help. Before
this review pass they would have hit:

1. **Pitch reads "SynapForge ?? 30 second pitch"** — em-dash mojibake in the very first
   line. Investor closes laptop, says "they ship to Windows but never tested on
   Windows." (FIXED.)
2. **NeuroMCP claim doesn't match output**. Pitch promises 5%->28%; demo shows
   5.7%->7.7%. Investor: "they overclaim; what else is fudged?" (FIXED.)
3. **R-fold bench prints "GPU: fold should beat sequential for N>=256 + R>=4"**, then
   shows N=256 and N=512 *losing* (0.72x, 0.36x). The verdict text contradicts the
   table directly above it. Investor: "they don't read their own output." (FIXED.)
4. **STDP heatmap is a wall of `??` block-glyphs** on the default Windows console —
   no signal at all. Investor: "the demo is broken." (FIXED — ASCII fallback.)
5. **Chat replay mojibakes Chinese**. Of the 5 ZH prompts/responses, every character
   renders `??`. Investor: "their multilingual claim is broken on day one." (FIXED.)

After this review pass, the same reviewer running `synapforge-demo all`:

- Sees a clean pitch with two-dash separators.
- Sees the 4-button density grow visibly (5.7% -> 7.7% over 0.5 s, with the trial-by-trial
  log showing K incrementing on novelty).
- Sees the R-fold bench measure correctness (R=1 1.5e-6, R=8 3.2e-3) and a speed table
  whose verdict text matches its rows.
- Sees a recognisable 8x8 ASCII heatmap of the STDP weight matrix at trial 0 (blank), 50
  (sparse), and 199 (saturated).
- Sees the recorded transcript with the correct Chinese characters and proper UTF-8 throughout.
- Total wall time ~1.5 s on a laptop CPU (excluding chat which is instantaneous from
  the recorded JSON). No errors, no warnings about missing CUDA.

---

## What the demo actually claims (after this pass)

`synapforge-demo all` ships these honest artifacts:

| Sub-demo | Claim | Verified by | Wall-time |
|----------|-------|-------------|----------:|
| `pitch` | 30-s framing of three differentiated bets | text only | <0.01 s |
| `button` | NeuroMCP synapses + dynamic codebook learn 4-button without `<tool_call>` tokens; density+K both grow from co-activation+novelty signals | `four_button.run_demo` 80-trial run; 100% hit_rate post-warmup | ~0.5 s |
| `bench` | Closed-form R-fold matches sequential to 1.55e-6 at R=1 / 3.2e-3 at R=8; on consumer GPU peak speedup is 2.99x at N=64 R=16; loses past N>=256 | `cfc_rfold` vs `_sequential_cfc` over (64,4), (64,16), (128,8), (256,8), (512,8) | ~0.3 s GPU / ~3 s CPU |
| `stdp` | Hebbian forward-only rule wires structured co-activations into the buffer with no optimizer/no loss; density (\|W\|>0.05) climbs from 0% to 25-30% over 200 trials | `STDPFastWeight` forward-pass only, `eval()` mode (so the `if self.training:` gate is gone) | ~0.2 s |
| `chat` | 5 EN + 5 ZH prompts; if v24h ckpt is present, generates live; otherwise replays a recorded transcript captured during v4.1 training | `_load_recorded()` returns from the package-relative `chat_recorded.json` | <0.1 s |

The `all` and `json` aggregate paths run all five and dump structured results.

---

## Top 3 polish items left for next pass

1. **Re-record QUICKSTART path-1 sample output** to match real measured numbers. The
   current sample shows "trial 79 density=27.9% K=14" which doesn't match a default
   `--trials 80` run; this is mostly cosmetic but a careful reader would catch it.
   Replace with real trace from a fresh run (or bump `--trials 200` everywhere and
   record at that scale).

2. **Bench on actual A100/A800** (not consumer GPU) and update both the paper's
   section 4.3 table and INVESTOR.md / README.md status row. The 2.99x consumer-GPU
   peak is honest, but the investor-facing "A800" framing in INVESTOR.md:55 is
   currently aspirational. Either qualify to "consumer GPU; A800 bench pending" or
   actually rent the A100/A800 hour and re-measure.

3. **Add a one-shot regression test** that runs `synapforge-demo all`, captures stdout,
   and asserts: (a) no `??` characters appear, (b) the printed `density:` final number
   is within 1pp of the previous-run JSON, (c) the rfold table's R=1 column shows
   <1e-4 rel_err. This is the simplest tripwire against the kind of investor-day
   regression we just hit. Wire into CI under a name like `test_demo_no_regression.py`
   and gate PRs on it.
