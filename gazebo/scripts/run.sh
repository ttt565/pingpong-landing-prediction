#!/usr/bin/env bash
# One serve in Gazebo Harmonic -> record landing -> run prediction pipeline.
# Linux host with Gazebo Harmonic (gz-sim8) required. See gazebo/README.md.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# 1) build the aero plugin (first run only)
if [ ! -f "$HERE/plugins/aero_launch/build/libAeroLaunch.so" ]; then
  cmake -S "$HERE/plugins/aero_launch" -B "$HERE/plugins/aero_launch/build"
  cmake --build "$HERE/plugins/aero_launch/build"
fi

# 2) make the plugin + model discoverable
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HERE/plugins/aero_launch/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HERE/models:${GZ_SIM_RESOURCE_PATH}"

# 3) start the recorder, then run the world headless for ~1 s of sim time
python3 "$HERE/scripts/record_landing.py" &
REC=$!
sleep 1
gz sim -s -r --iterations 1000 "$HERE/worlds/table_tennis.sdf"
wait $REC

# 4) feed Gazebo truth into the SAME prediction pipeline
python3 "$HERE/scripts/predict_from_csv.py" traj.csv landing.csv --fps 120 --bad_frac 0.2
