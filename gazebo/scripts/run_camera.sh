#!/usr/bin/env bash
# Route B: one serve with two rendered cameras -> pixel detection -> stereo
# triangulation -> the SAME prediction pipeline on real rendered noise.
#
#   bash run_camera.sh          # sharp: 120fps cameras, no net  (H~0 null)
#   bash run_camera.sh blur     # realism pack: 500Hz renders blended into
#                               # 125fps full-shutter frames (motion blur) +
#                               # the net occluding the back camera near landing
#
# Requires headless rendering (ogre2/EGL). If it fails on your GPU stack, try:
#   export LIBGL_ALWAYS_SOFTWARE=1
set -e
MODE="${1:-sharp}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/scripts"

if [ "$MODE" = "blur" ]; then
  WORLD="$HERE/worlds/table_tennis_cam_blur.sdf"
  OUT="$HERE/cam_blur_out"
  REPORT="$HERE/results_camera_blur.md"
  BLEND=4
else
  WORLD="$HERE/worlds/table_tennis_cam.sdf"
  OUT="$HERE/cam_out"
  REPORT="$HERE/results_camera.md"
  BLEND=1
fi
mkdir -p "$OUT"

if [ ! -f "$HERE/plugins/aero_launch/build/libAeroLaunch.so" ]; then
  cmake -S "$HERE/plugins/aero_launch" -B "$HERE/plugins/aero_launch/build"
  cmake --build "$HERE/plugins/aero_launch/build"
fi
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HERE/plugins/aero_launch/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HERE/models:${GZ_SIM_RESOURCE_PATH}"

python3 record_landing.py --outdir "$OUT" --bounces 1 --timeout 300 &
REC=$!
python3 camera_track.py --outdir "$OUT" --blend "$BLEND" --duration 300 &
CAM=$!
sleep 1

# 0.5 s sim time covers the full pre-landing arc; rendering may run well below
# real time, which is fine — everything is stamped in sim time.
gz sim -s -r --headless-rendering --iterations 500 "$WORLD"

wait $REC
wait $CAM

python3 camera_predict.py "$OUT" --out "$REPORT"
