# Route B: stereo-camera perception -> prediction

36 stereo frames @120 fps (640x480, hfov 1.2), stereo 3D RMS vs truth = 5.3 mm, truth landing x=1.7531 y=0.0000 m

| method | landing error (cm) |
|---|---|
| M0 | 6.24 |
| M1 | 0.53 |
| M3_conf | 0.55 |

Noise here is real rendering/quantization noise — roughly homoscedastic per arc, so M3 ≈ M1 is the EXPECTED outcome (that is the H≈0 null of the killer experiment, measured on rendered pixels instead of assumed).
