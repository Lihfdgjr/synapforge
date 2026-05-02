# Data Quality Report

_Generated UTC: 2026-05-02 12:20:31_

Token-soup root-cause analysis 2026-05-02. Higher composite score = better. The trainer's quality gate **PASSes** when all four hard thresholds are met:

    top1_share  < 0.10
    vocab_cov   > 0.30
    bigram_rep  < 0.05
    uniq/100    > 0.40

## Summary table (sorted by composite, descending)

| File | rows | tokens | top1 | top5 | vocab_cov | uniq/100 | bigram_rep | median_len | composite | gate |
|------|------|--------|------|------|-----------|----------|------------|------------|-----------|------|
| `000_00000.parquet` | 8,000 | 7,425,765 | 0.048 | 0.194 | 1.000 | 0.693 | 0.0301 | 536 | 0.847 | PASS |
| `train-00000.parquet` | 8,000 | 6,340,244 | 0.156 | 0.335 | 1.000 | 0.573 | 0.0296 | 336 | 0.815 | FAIL |
| `0011.parquet` | 2,038 | 66,259 | 0.058 | 0.173 | 0.268 | 0.000 | 0.0190 | 35 | 0.589 | FAIL |
| `0003.parquet` | 2,039 | 73,682 | 0.064 | 0.177 | 0.278 | 0.000 | 0.0231 | 37 | 0.577 | FAIL |
| `0007.parquet` | 2,038 | 71,176 | 0.061 | 0.172 | 0.288 | 0.000 | 0.0258 | 37 | 0.572 | FAIL |
| `0005.parquet` | 2,039 | 71,745 | 0.064 | 0.177 | 0.280 | 0.000 | 0.0246 | 37 | 0.572 | FAIL |
| `0008.parquet` | 2,038 | 72,312 | 0.061 | 0.173 | 0.261 | 0.000 | 0.0249 | 37 | 0.561 | FAIL |
| `0006.parquet` | (err: ssh fail 255: Timeout, server mohuanfang) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | FAIL |

## Per-file detail

### `/home/liu/lnn-train/data/fineweb_edu/000_00000.parquet`

- text_column: `text`
- rows_sampled: 8,000
- tokens_total: 7,425,765
- composite_score: **0.847**  (quality_gate: **PASS**)

**Token frequency**
- top-1 token share: 0.0477  (target < 0.10)
- top-5 token share: 0.1943
- top-50 token share: 0.4182
- vocab coverage: 1.0000  (target > 0.30)
- top-10 tokens: `44073020`=354,004 (4.77%), `111298773`=353,166 (4.76%), `65220392`=345,256 (4.65%), `905156924`=212,967 (2.87%), `219662775`=177,279 (2.39%), `474440566`=155,574 (2.10%), `634337033`=118,607 (1.60%), `438350758`=116,002 (1.56%), `414585068`=99,754 (1.34%), `48944103`=74,151 (1.00%)

**Sequence stats**
- avg seq_len: 928.2, median: 536.0, p95: 2889

**Readability**
- unique-token-per-100 ratio: 0.6934  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0301  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.796  digit: 0.011  punct: 0.029  CJK: 0.000

### `/home/liu/lnn-train/data/wiki_zh/train-00000.parquet`

- text_column: `text`
- rows_sampled: 8,000
- tokens_total: 6,340,244
- composite_score: **0.815**  (quality_gate: **FAIL**)

**Token frequency**
- top-1 token share: 0.1557  (target < 0.10)
- top-5 token share: 0.3345
- top-50 token share: 0.4962
- vocab coverage: 1.0000  (target > 0.30)
- top-10 tokens: `720217265`=987,142 (15.57%), `70928126`=524,370 (8.27%), `320966800`=277,229 (4.37%), `996787102`=166,082 (2.62%), `1008369848`=165,818 (2.62%), `564373673`=113,638 (1.79%), `938880819`=74,438 (1.17%), `515018429`=74,434 (1.17%), `1060057925`=72,009 (1.14%), `928519153`=67,177 (1.06%)

**Sequence stats**
- avg seq_len: 792.5, median: 336.0, p95: 3000

**Readability**
- unique-token-per-100 ratio: 0.5726  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0296  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.061  digit: 0.061  punct: 0.011  CJK: 0.725

### `/home/liu/lnn-train/data/librispeech/train-clean-100/0011.parquet`

- text_column: `text`
- rows_sampled: 2,038
- tokens_total: 66,259
- composite_score: **0.589**  (quality_gate: **FAIL**)

**Token frequency**
- top-1 token share: 0.0577  (target < 0.10)
- top-5 token share: 0.1733
- top-50 token share: 0.4503
- vocab coverage: 0.2676  (target > 0.30)
- top-10 tokens: `498329114`=3,826 (5.77%), `747075214`=2,411 (3.64%), `798486782`=2,134 (3.22%), `562694892`=1,797 (2.71%), `429158409`=1,317 (1.99%), `778863238`=1,076 (1.62%), `587819534`=1,068 (1.61%), `937148445`=796 (1.20%), `171329314`=792 (1.20%), `795669150`=757 (1.14%)

**Sequence stats**
- avg seq_len: 32.5, median: 35.0, p95: 50

**Readability**
- unique-token-per-100 ratio: 0.0000  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0190  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.817  digit: 0.000  punct: 0.001  CJK: 0.000

### `/home/liu/lnn-train/data/librispeech/train-clean-100/0003.parquet`

- text_column: `text`
- rows_sampled: 2,039
- tokens_total: 73,682
- composite_score: **0.577**  (quality_gate: **FAIL**)

**Token frequency**
- top-1 token share: 0.0635  (target < 0.10)
- top-5 token share: 0.1766
- top-50 token share: 0.4476
- vocab coverage: 0.2779  (target > 0.30)
- top-10 tokens: `905637714`=4,680 (6.35%), `462153201`=2,422 (3.29%), `958887213`=2,318 (3.15%), `72102018`=1,986 (2.70%), `407500862`=1,604 (2.18%), `503798100`=1,282 (1.74%), `283521927`=957 (1.30%), `399635839`=942 (1.28%), `531890765`=881 (1.20%), `92841382`=826 (1.12%)

**Sequence stats**
- avg seq_len: 36.1, median: 37, p95: 53

**Readability**
- unique-token-per-100 ratio: 0.0000  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0231  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.817  digit: 0.000  punct: 0.002  CJK: 0.000

### `/home/liu/lnn-train/data/librispeech/train-clean-100/0007.parquet`

- text_column: `text`
- rows_sampled: 2,038
- tokens_total: 71,176
- composite_score: **0.572**  (quality_gate: **FAIL**)

**Token frequency**
- top-1 token share: 0.0609  (target < 0.10)
- top-5 token share: 0.1715
- top-50 token share: 0.4402
- vocab coverage: 0.2882  (target > 0.30)
- top-10 tokens: `465701380`=4,333 (6.09%), `284347234`=2,257 (3.17%), `167778625`=2,199 (3.09%), `32156718`=1,823 (2.56%), `953309023`=1,597 (2.24%), `676194291`=1,267 (1.78%), `368067257`=941 (1.32%), `721925409`=792 (1.11%), `539872032`=779 (1.09%), `744547345`=736 (1.03%)

**Sequence stats**
- avg seq_len: 34.9, median: 37.0, p95: 51

**Readability**
- unique-token-per-100 ratio: 0.0000  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0258  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.818  digit: 0.000  punct: 0.002  CJK: 0.000

### `/home/liu/lnn-train/data/librispeech/train-clean-100/0005.parquet`

- text_column: `text`
- rows_sampled: 2,039
- tokens_total: 71,745
- composite_score: **0.572**  (quality_gate: **FAIL**)

**Token frequency**
- top-1 token share: 0.0641  (target < 0.10)
- top-5 token share: 0.1772
- top-50 token share: 0.4449
- vocab coverage: 0.2797  (target > 0.30)
- top-10 tokens: `1059598564`=4,596 (6.41%), `357906282`=2,399 (3.34%), `93083829`=2,215 (3.09%), `632291468`=1,908 (2.66%), `102238130`=1,598 (2.23%), `197575269`=1,128 (1.57%), `153071913`=1,120 (1.56%), `342648884`=1,021 (1.42%), `415773587`=869 (1.21%), `546865392`=773 (1.08%)

**Sequence stats**
- avg seq_len: 35.2, median: 37, p95: 52

**Readability**
- unique-token-per-100 ratio: 0.0000  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0246  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.819  digit: 0.000  punct: 0.002  CJK: 0.000

### `/home/liu/lnn-train/data/librispeech/train-clean-100/0008.parquet`

- text_column: `text`
- rows_sampled: 2,038
- tokens_total: 72,312
- composite_score: **0.561**  (quality_gate: **FAIL**)

**Token frequency**
- top-1 token share: 0.0614  (target < 0.10)
- top-5 token share: 0.1727
- top-50 token share: 0.4608
- vocab coverage: 0.2606  (target > 0.30)
- top-10 tokens: `1068454952`=4,439 (6.14%), `583144829`=2,394 (3.31%), `665144580`=2,192 (3.03%), `560421731`=1,932 (2.67%), `761937999`=1,528 (2.11%), `1015515823`=1,312 (1.81%), `495461040`=1,143 (1.58%), `117933062`=1,137 (1.57%), `951232658`=1,039 (1.44%), `857474933`=966 (1.34%)

**Sequence stats**
- avg seq_len: 35.5, median: 37.0, p95: 52

**Readability**
- unique-token-per-100 ratio: 0.0000  (target 0.40 < x < 0.75; sweet spot 0.55-0.65)
- bigram repetition (50-tok windows): 0.0249  (target < 0.05)

**Char distribution**
- alpha (ascii): 0.814  digit: 0.000  punct: 0.002  CJK: 0.000

### `myserver:/home/liu/lnn-train/data/librispeech/train-clean-100/0006.parquet`

**ERROR**: ssh fail 255: Timeout, server mohuanfang.com not responding.



