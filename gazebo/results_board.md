# Contact-board self-supervision: residual learning

114 valid serves from a 3-cluster serve repertoire (topspin/flat/backspin), one perception-noise realization per serve, physics pipeline = M3_conf fit -> calibrated bounce -> flight to the board plane (x = 2.78 m). Test set: 24 held-out serves; ridge on 9 features, lambda=1.

## Board-contact error (cm, mean over test serves)

| model | error |
|---|---|
| physics only | 20.8 |
| physics + ridge residual (full pool, n=90) | 3.9 |
| true-spin oracle (ceiling) | 6.8 |

## Sample efficiency (test error vs labeled serves)

| train serves | error (cm) mean ± std over subsets |
|---|---|
| 10 | 7.3 ± 2.5 |
| 20 | 5.1 ± 0.6 |
| 40 | 4.2 ± 0.2 |
| 80 | 3.9 ± 0.1 |

## Per-cluster (full-pool model, test set)

| cluster | physics | + learned | oracle |
|---|---|---|---|
| topspin | 25.9 | 3.0 | 6.7 |
| flat | 22.3 | 4.5 | 7.6 |
| backspin | 10.3 | 4.7 | 6.2 |
