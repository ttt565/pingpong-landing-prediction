# Real-video exit test (Phase-1 go/no-go measurement)

The killer experiment's verdict (see `../LIMITATIONS.md`, robust-baseline
section) says confidence weighting is worth deploying only if the real
detector's confidence passes the `results/fig5_miscalibration.png` thresholds:

- calibration exponent **γ ≥ ~0.5** (confidence actually tracks precision),
- confidence log-noise **≤ ~0.6** (past ~1.0 confidence weighting *hurts*),

and even then the M3-vs-robust-loss gap only opens at high bad-frame rates.
`exit_test.py` measures exactly these numbers — plus within-arc H and the
dropout rate — on **OpenTTGames** (120 fps static-camera table tennis with
per-frame ball labels; CC BY-NC-SA 4.0, research use).

## Data (not committed, ~226 MB)

```bash
mkdir -p data && cd data
curl -L -O https://lab.osai.ai/datasets/openttgames/data/test_2.mp4
curl -L -o test_2.zip https://lab.osai.ai/datasets/openttgames/data/test_2.zip
unzip test_2.zip -d test_2_markup
```

## Run

```bash
python3 exit_test.py                      # built-in detector
python3 exit_test.py --detections my.csv  # score ANY detector (frame,u,v,conf)
```

The built-in detector is the same class as the Gazebo Route-B one
(temporal-median background subtraction + small-blob temporal continuity) — a
lightweight stand-in for TrackNet. The **harness is detector-agnostic**: run
TrackNet/TTNet offline, dump `frame,u,v,conf` to CSV, and score it with
`--detections` to get the same verdict for the production detector.
`detections.csv` (committed) holds the built-in detector's output so the
analysis is reproducible without downloading the video.

Fair-protocol note: detection uses no labels except a disclosed one-time
anchor at each arc's first labeled frame (a real system anchors on serve
detection). Labels themselves come from OpenTTGames' deep-learning-aided
annotation, so "error" is relative to that reference, not absolute truth.

Results: [results_exit_test.md](results_exit_test.md) + fig_exit_test.png.
