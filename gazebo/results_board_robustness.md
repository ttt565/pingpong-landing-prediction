# Contact-board learning: label noise + online RLS

Same dataset/protocol as results_board.md (114 serves, 24 held-out; physics only = 20.8 cm, true-spin oracle = 6.8 cm). Train labels get Gaussian plate noise; evaluation is always against the true contact.

## 1. Test error (cm) vs training-label noise

| train serves | σ=0 cm | σ=0.5 cm | σ=1 cm | σ=2 cm | σ=3 cm | σ=5 cm |
|---|---|---|---|---|---|---|
| 10 | 7.0 ± 1.4 | 7.0 ± 1.6 | 7.0 ± 1.5 | 7.1 ± 1.3 | 8.2 ± 1.7 | 9.0 ± 2.1 |
| 20 | 5.1 ± 0.7 | 5.2 ± 0.7 | 5.3 ± 0.6 | 5.6 ± 0.7 | 5.9 ± 0.9 | 6.8 ± 1.2 |
| 40 | 4.3 ± 0.3 | 4.3 ± 0.3 | 4.3 ± 0.3 | 4.4 ± 0.4 | 4.6 ± 0.5 | 5.2 ± 0.5 |
| 80 | 3.9 ± 0.1 | 3.9 ± 0.1 | 4.0 ± 0.1 | 4.0 ± 0.2 | 4.1 ± 0.3 | 4.5 ± 0.5 |

## 2. Online RLS, prequential error (cm) at serve #k

| update-label noise | k=10 | k=20 | k=50 | k=100 | last-20 mean |
|---|---|---|---|---|---|
| σ=0 cm | 7.2 | 5.0 | 4.4 | 3.9 | 3.6 |
| σ=2 cm | 8.1 | 6.1 | 4.1 | 3.6 | 3.7 |
