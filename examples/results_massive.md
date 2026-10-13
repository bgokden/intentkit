### MASSIVE, 8-shot, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

60 intents, 500 parallel test utterances per language, closed-set top-1 accuracy. SetFit numbers are the mean of 2 seeds.

| Training data | Method | Train size | en | de | es | fr | tr | **Mean** | Fit |
|---|---|---|---|---|---|---|---|---|---|
| en-only | kNN (no training) | 476 | 0.608 | 0.458 | 0.526 | 0.546 | 0.512 | **0.530** | 6s |
| en-only | SetFit | 476 | 0.662 | 0.480 | 0.591 | 0.591 | 0.557 | **0.576** | 143s |
| en + 2/lang | kNN (no training) | 956 | 0.630 | 0.478 | 0.548 | 0.572 | 0.528 | **0.551** | 4s |
| en + 2/lang | SetFit | 956 | 0.724 | 0.626 | 0.674 | 0.691 | 0.651 | **0.673** | 247s |

### MASSIVE, 16-shot, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

60 intents, 500 parallel test utterances per language, closed-set top-1 accuracy. SetFit numbers are the mean of 1 seed.

| Training data | Method | Train size | en | de | es | fr | tr | **Mean** | Fit |
|---|---|---|---|---|---|---|---|---|---|
| en-only | kNN (no training) | 946 | 0.656 | 0.476 | 0.564 | 0.576 | 0.566 | **0.568** | 5s |
| en-only | SetFit | 946 | 0.726 | 0.532 | 0.636 | 0.660 | 0.606 | **0.632** | 236s |
| en + 2/lang | kNN (no training) | 1426 | 0.654 | 0.504 | 0.564 | 0.582 | 0.566 | **0.574** | 4s |
| en + 2/lang | SetFit | 1426 | 0.740 | 0.650 | 0.694 | 0.714 | 0.670 | **0.694** | 358s |

