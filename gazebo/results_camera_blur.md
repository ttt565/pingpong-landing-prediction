# Route B: stereo-camera perception -> prediction

32 stereo frames @120 fps (640x480, hfov 1.2), stereo 3D RMS vs truth = 5.8 mm, truth landing x=1.7531 y=0.0000 m

| method | landing error (cm) |
|---|---|
| M0 | 8.68 |
| M1 | 1.47 |
| M3_conf | 1.45 |

Noise here is real rendering noise (quantization; in blur mode also exposure smear + net occlusion). Within-arc it stays near-homoscedastic — hard failures drop out instead of degrading — so M3 ≈ M1 is the EXPECTED outcome: the killer experiment's H≈0 null, measured on pixels. M3's regime (flaggable bad detections) needs detector-level confusion, which pure rendering does not produce.
