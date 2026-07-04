# Real-video exit test (OpenTTGames test_2, 120 fps)

697 scored detections over 35 arcs; dropout 9.4% of labeled frames, gross failures (err>60 px) 13.5%. Detector: median-background + temporal continuity (lightweight TrackNet stand-in; harness accepts any detector CSV).

## Confidence-decile calibration

| bin | median conf | sigma (px) |
|---|---|---|
| 0 | 52.0 | 3.56 |
| 1 | 75.2 | 2.19 |
| 2 | 94.7 | 1.72 |
| 3 | 107.5 | 1.35 |
| 4 | 125.3 | 1.38 |
| 5 | 139.2 | 1.29 |
| 6 | 156.7 | 1.27 |
| 7 | 187.5 | 1.22 |

## Exit-test numbers vs the fig5 thresholds

- sigma dynamic range across confidence bins: 2.93x
- calibration exponent gamma-hat = 1.06 (threshold from run_miscalibration: >= 0.5)
- confidence log-noise cn-hat = 0.17 (threshold: <= ~0.6; at >= 1.0 confidence weighting HURTS)
- within-arc heteroscedasticity H = 0.31 (mean over arcs; sweep says the M3-vs-robust gap only opens at high bad-frame rates even when H ~ 1)
- dropout rate 9.4% — failures are dropouts, as in the rendered Route B

## Verdict
gamma-hat passes and cn-hat passes the fig5 thresholds for this detector on this footage. Per the robust-baseline result, the deployable default remains a robust loss; confidence weighting is justified only if BOTH pass on the production detector.
