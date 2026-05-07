#!/usr/bin/env bash
set -eo pipefail

source_if_exists() {
  local target="$1"
  if [ ! -f "$target" ]; then
    echo "[rover_entrypoint] Required file not found: $target" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$target"
}

source_if_exists "/home/docker/venv/bin/activate"
source_if_exists "/opt/ros/${ROS_DISTRO}/setup.bash"
source_if_exists "/opt/dave_ws/install/setup.bash"
source_if_exists "${SAILBOAT_WS}/install/setup.bash"

set -u

export PATH="/home/docker/ardupilot_ws/ardupilot/Tools/autotest:${PATH}"
export PATH="/home/docker/ardupilot_ws/ardupilot/build/sitl/bin:${PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/asv_sim_ws/install/lib:/home/docker/ardupilot_ws/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH}:/opt/dave_ws/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib"
export GZ_SIM_RESOURCE_PATH="/opt/asv_sim_ws/src/asv_sim/asv_sim_gazebo/models:/opt/asv_sim_ws/src/asv_sim/asv_sim_gazebo/worlds:/home/docker/ardupilot_ws/ardupilot_gazebo/models:/home/docker/ardupilot_ws/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export LD_LIBRARY_PATH="/opt/dave_ws/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib:${LD_LIBRARY_PATH:-}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-docker}"

START_QGC="${START_QGC:-1}"
ROBOT_LAUNCH_PACKAGE="${ROBOT_LAUNCH_PACKAGE:-dave_demos}"
ROBOT_LAUNCH_FILE="${ROBOT_LAUNCH_FILE:-dave_robot.launch.py}"
ROBOT_LAUNCH_ARGS="${ROBOT_LAUNCH_ARGS:-namespace:=sailboat world_name:=waves_wind}"

RESTART_LAUNCH="${RESTART_LAUNCH:-1}"
RESTART_DELAY="${RESTART_DELAY:-2}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

launch_pid=""
stop_requested=0

start_qgc() {
  if [ "${START_QGC}" != "1" ]; then
    return
  fi

  if ! command -v qgroundcontrol >/dev/null 2>&1; then
    echo "[rover_entrypoint] qgroundcontrol command not found" >&2
    return
  fi

  if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    echo "[rover_entrypoint] QGC autostart skipped: no DISPLAY/WAYLAND_DISPLAY" >&2
    return
  fi

  if pgrep -f qgroundcontrol >/dev/null 2>&1; then
    return
  fi

  nohup bash -lc 'unset WAYLAND_DISPLAY; export QT_QPA_PLATFORM=xcb; export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-docker}"; qgroundcontrol' >/tmp/qgc.log 2>&1 &
}

forward_signal() {
  stop_requested=1

  if [ -n "${launch_pid}" ] && kill -0 "${launch_pid}" >/dev/null 2>&1; then
    kill -TERM "${launch_pid}" >/dev/null 2>&1 || true
  fi
}

run_default_launch() {
  local attempt=1
  local exit_code=0
  local -a cmd
  local -a extra_args=()

  trap forward_signal INT TERM

  while true; do
    echo "[rover_entrypoint] Starting default launch (attempt ${attempt})"

    cmd=(ros2 launch "${ROBOT_LAUNCH_PACKAGE}" "${ROBOT_LAUNCH_FILE}")

    if [ -n "${ROBOT_LAUNCH_ARGS}" ]; then
      read -r -a extra_args <<< "${ROBOT_LAUNCH_ARGS}"
      cmd+=("${extra_args[@]}")
    fi

    "${cmd[@]}" &
    launch_pid=$!

    start_qgc

    set +e
    wait "${launch_pid}"
    exit_code=$?
    set -e

    launch_pid=""

    if [ "${stop_requested}" -eq 1 ]; then
      exit "${exit_code}"
    fi

    if [ "${RESTART_LAUNCH}" != "1" ]; then
      exit "${exit_code}"
    fi

    if [ "${MAX_RESTARTS}" -gt 0 ] && [ "${attempt}" -ge "${MAX_RESTARTS}" ]; then
      echo "[rover_entrypoint] Launch exited with code ${exit_code}; reached restart limit (${MAX_RESTARTS})" >&2
      exit "${exit_code}"
    fi

    echo "[rover_entrypoint] Launch exited with code ${exit_code}; restarting in ${RESTART_DELAY}s" >&2
    sleep "${RESTART_DELAY}"
    attempt=$((attempt + 1))
  done
}

if [ "$#" -eq 0 ]; then
  run_default_launch
  exit $?
fi

exec "$@"