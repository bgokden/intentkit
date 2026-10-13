### `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Test set: 110 queries (18 in-scope + 4 out-of-scope per language). SetFit numbers are the mean of 3 seeds. Fit time on CPU (Apple M-series).

| Training data | Method | en | de | es | fr | tr | **Overall** | In-scope acc | Closed-set acc | OOS recall | Fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| en-only | kNN (no training) | 1.00 | 0.91 | 0.91 | 0.95 | 0.95 | **0.95** | 0.98 | 0.98 | 0.80 | 5s |
| en-only | SetFit | 0.98 | 0.88 | 0.94 | 0.97 | 0.88 | **0.93** | 0.96 | 0.97 | 0.82 | 18s |
| en-only | SetFit + en negatives | 0.95 | 0.77 | 0.86 | 0.85 | 0.76 | **0.84** | 0.80 | 0.97 | 1.00 | 23s |
| en-only | SetFit + multilingual negatives | 0.98 | 0.76 | 0.91 | 0.80 | 0.77 | **0.85** | 0.81 | 0.93 | 1.00 | 25s |
| en + 2/lang | kNN (no training) | 1.00 | 0.91 | 1.00 | 1.00 | 1.00 | **0.98** | 0.98 | 0.99 | 1.00 | 3s |
| en + 2/lang | SetFit | 0.97 | 0.89 | 0.92 | 0.92 | 0.94 | **0.93** | 0.99 | 0.99 | 0.67 | 29s |
| en + 2/lang | SetFit + en negatives | 0.95 | 0.88 | 0.94 | 0.94 | 0.88 | **0.92** | 0.91 | 0.99 | 0.97 | 35s |
| en + 2/lang | SetFit + multilingual negatives | 1.00 | 0.92 | 0.98 | 1.00 | 0.98 | **0.98** | 0.99 | 0.99 | 0.95 | 37s |

