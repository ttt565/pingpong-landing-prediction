#!/usr/bin/env bash
# Route B: one serve with two rendered 120fps cameras -> pixel detection ->
# stereo triangulation -> the SAME prediction pipeline on real rendered noise.
# Requires headless rendering (ogre2/EGL). If it fails on your GPU stack, try:
#   export LIBGL_ALWAYS_SOFTWARE=1
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/scripts"
OUT="$HERE/cam_out"
mkdir -p "$OUT"

if [ ! -f "$HERE/plugins/aero_launch/build/libAeroLaunch.so" ]; then
  cmake -S "$HERE/plugins/aero_launch" -B "$HERE/plugins/aero_launch/build"
  cmake --build "$HERE/plugins/aero_launch/build"
fi
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HERE/plugins/aero_launch/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HERE/models:${GZ_SIM_RESOURCE_PATH}"

python3 record_landing.py --outdir "$OUT" --bounces 1 --timeout 180 &
REC=$!
python3 camera_track.py --outdir "$OUT" --duration 180 &
CAM=$!
sleep 1

# 0.5 s sim time covers the full pre-landing arc; rendering may run well below
# real time, which is fine — everything is stamped in sim time.
gz sim -s -r --headless-rendering --iterations 500 "$HERE/worlds/table_tennis_cam.sdf"

wait $REC
wait $CAM

python3 camera_predict.py "$OUT" --out "$HERE/results_camera.md"
