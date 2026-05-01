# R-fold outreach package

Three artifacts and a reproduction script for the R-fold note. Honest tone, no marketing
fluff. Submit each one independently, or sequentially over a single day.

## Contents

| File | Length | Purpose |
|---|---:|---|
| `../docs/RFOLD_PAPER.md` | ~600 LOC | Paper draft (markdown), ready for arXiv conversion |
| `../scripts/rfold_paper_repro.sh` | ~50 LOC | Single bash entrypoint for all paper numbers |
| `twitter_thread.md` | 8 tweets | X / Twitter thread, edit before posting |
| `hn_post.md` | ~150 LOC | Hacker News Show HN body |
| `README.md` | this file | submission checklist |

## Pre-submission checklist

- [ ] `bash scripts/rfold_paper_repro.sh` runs cleanly on a fresh clone
- [ ] `paper_repro/rfold_correctness.json` shows R=1 < 1e-4 and R=8 < 0.05
- [ ] `paper_repro/rfold_speed_gpu.json` is present (skip-marker is OK if no CUDA on
      author's machine but a GPU number must be in the paper, run on rented hw)
- [ ] All four references (2401.13386, 2208.04933, 2412.06769, 2410.10841) are linked in
      the paper and resolve via `arxiv.org/abs/<id>`
- [ ] No "167x" claim survives anywhere outside Appendix B (the methodological note)
- [ ] Twitter thread fits 8-10 tweets, each under 280 chars
- [ ] HN post is a Show HN with the GitHub link in the URL field, not just the body
- [ ] Author handle on X / GitHub matches the paper attribution
- [ ] License (LICENSE in repo root) is permissive enough for the audience

## Submission order

1. Push the repo updates (paper, repro script).
2. Soft-launch on X with the thread above. Record reactions for 24h.
3. If the X thread gets traction, post the HN Show HN. Otherwise, file as draft.
4. Optional: convert `RFOLD_PAPER.md` to LaTeX and submit to arXiv (cs.LG, cs.NE
   secondary).
