**All 498 scorable questions** (of 918; yes/no-style answers excluded)

| Chunk size | Overlap | Chunks | hit@1 | hit@2 | hit@4 | hit@8 |
|---|---|---|---|---|---|---|
| 100 | 0 | 14,416 | 22.3% | 26.1% | 31.5% | 34.9% |
| 250 | 25 | 7,050 | 31.1% | 39.8% | 46.6% | 50.2% |
| 500 | 50 | 4,492 | 32.5% | 41.6% | 46.4% | 52.2% |
| 1000 | 100 | 3,405 | 33.5% | 42.4% | 47.8% | 53.0% |

**Answers of at most 100 characters only (471 questions)** — the subset that physically fits inside every chunk size, so no row is penalised for chunks too small to ever contain the gold answer

| Chunk size | Overlap | Chunks | hit@1 | hit@2 | hit@4 | hit@8 |
|---|---|---|---|---|---|---|
| 100 | 0 | 14,416 | 23.6% | 27.6% | 33.3% | 36.9% |
| 250 | 25 | 7,050 | 32.1% | 40.6% | 47.8% | 51.6% |
| 500 | 50 | 4,492 | 33.3% | 42.9% | 47.3% | 53.3% |
| 1000 | 100 | 3,405 | 34.4% | 43.7% | 48.8% | 53.9% |
