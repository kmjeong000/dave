#!/usr/bin/env bash
set -eo pipefail

VENV_SETUP="${VENV_SETUP:-/home/docker/.venv/bin/activate}"
DAVE_UNDERLAY="${DAVE_UNDERLAY:-/opt/dave_ws}"
APP_WS="${APP_WS:-/home/docker/app_ws}"
ARDUPILOT_WS="${ARDUPILOT_WS:-/home/docker/ardupilot_ws}"

source "${VENV_SETUP}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${DAVE_UNDERLAY}/install/setup.bash"
source "${APP_WS}/install/setup.bash"

set -u

WAVE_PLUGIN_PATH="${DAVE_UNDERLAY}/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/install/wave/lib"
WAVE_GUI_PLUGIN_PATH="${DAVE_UNDERLAY}/src/dave/gazebo/dave_gz_world_plugins/ocean-waves/src/gui/plugins/waves_control/build"

export GEOGRAPHICLIB_GEOID_PATH="${GEOGRAPHICLIB_GEOID_PATH:-/usr/local/share/GeographicLib/geoids}"
export PYTHONPATH="/opt/gazebo/install/lib/python:${PYTHONPATH:-}"
export PATH="${ARDUPILOT_WS}/ardupilot/Tools/autotest:${PATH}"
export PATH="${ARDUPILOT_WS}/ardupilot/build/sitl/bin:${PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${ARDUPILOT_WS}/ardupilot_gazebo/build:${WAVE_PLUGIN_PATH}:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_GUI_PLUGIN_PATH="${WAVE_GUI_PLUGIN_PATH}:${GZ_GUI_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="${ARDUPILOT_WS}/ardupilot_gazebo/models:${ARDUPILOT_WS}/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export LD_LIBRARY_PATH="${WAVE_PLUGIN_PATH}:${LD_LIBRARY_PATH:-}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

QGC_APPIMAGE="${QGC_APPIMAGE:-${HOME}/qgc/qgc.AppImage}"
START_QGC="${START_QGC:-0}"
GZ_SERVER_ONLY="${GZ_SERVER_ONLY:-1}"
VJOY_URI="${VJOY_URI:-}"
START_VJOY_HTML="${START_VJOY_HTML:-0}"
VJOY_HTML_DELAY="${VJOY_HTML_DELAY:-4}"
VJOY_BROWSER="${VJOY_BROWSER:-firefox}"

SAILDRONE_AUTOSTART="${SAILDRONE_AUTOSTART:-sim}"
SAILDRONE_RESTART_LAUNCH="${SAILDRONE_RESTART_LAUNCH:-0}"
SAILDRONE_RESTART_DELAY="${SAILDRONE_RESTART_DELAY:-2}"
SAILDRONE_MAX_RESTARTS="${SAILDRONE_MAX_RESTARTS:-0}"
SAILDRONE_LAUNCH_PACKAGE="${SAILDRONE_LAUNCH_PACKAGE:-dave_demos}"
SAILDRONE_LAUNCH_FILE="${SAILDRONE_LAUNCH_FILE:-dave_robot.launch.py}"
SAILDRONE_NAMESPACE="${SAILDRONE_NAMESPACE:-sailboat}"
SAILDRONE_WORLD_NAME="${SAILDRONE_WORLD_NAME:-waves_wind}"
SAILDRONE_PAUSED="${SAILDRONE_PAUSED:-false}"
SAILDRONE_GUI="${SAILDRONE_GUI:-true}"
SAILDRONE_USE_SIM_TIME="${SAILDRONE_USE_SIM_TIME:-true}"
SAILDRONE_USE_NED_FRAME="${SAILDRONE_USE_NED_FRAME:-false}"
SAILDRONE_X="${SAILDRONE_X:-0.0}"
SAILDRONE_Y="${SAILDRONE_Y:-0.0}"
SAILDRONE_Z="${SAILDRONE_Z:-0.0}"
SAILDRONE_ROLL="${SAILDRONE_ROLL:-0.0}"
SAILDRONE_PITCH="${SAILDRONE_PITCH:-0.0}"
SAILDRONE_YAW="${SAILDRONE_YAW:-0.0}"

launch_pid=""
qgc_pid=""
stop_requested=0

start_qgc() {
  if [ "${START_QGC}" != "1" ]; then
    return 0
  fi

  if [ ! -x "${QGC_APPIMAGE}" ]; then
    echo "[saildrone_entrypoint] QGC AppImage not found or not executable: ${QGC_APPIMAGE}" >&2
    return 1
  fi

  if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    echo "[saildrone_entrypoint] QGC autostart skipped: no DISPLAY/WAYLAND_DISPLAY" >&2
    return 1
  fi

  if pgrep -f "${QGC_APPIMAGE}" >/dev/null 2>&1; then
    return 0
  fi

  nohup "${QGC_APPIMAGE}" >/tmp/qgc.log 2>&1 &
  qgc_pid=$!
}

open_uri() {
  local uri="$1"

  if command -v "${VJOY_BROWSER}" >/dev/null 2>&1; then
    "${VJOY_BROWSER}" --new-tab "${uri}"
    return
  fi

  if command -v firefox >/dev/null 2>&1; then
    firefox --new-tab "${uri}"
    return
  fi

  if command -v gio >/dev/null 2>&1; then
    gio open "${uri}"
    return
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${uri}"
    return
  fi

  if command -v sensible-browser >/dev/null 2>&1; then
    sensible-browser "${uri}"
    return
  fi

  python3 -m webbrowser "${uri}"
}

start_vjoy_html() {
  local uri="${VJOY_URI}"

  if [ "${START_VJOY_HTML}" != "1" ]; then
    return
  fi

  if [ -z "${uri}" ]; then
    echo "[saildrone_entrypoint] Virtual joystick autostart skipped: VJOY_URI is empty" >&2
    return
  fi

  if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    echo "[saildrone_entrypoint] Virtual joystick autostart skipped: no DISPLAY/WAYLAND_DISPLAY" >&2
    return
  fi

  if [[ "${uri}" != *"://"* ]]; then
    if [ ! -f "${uri}" ]; then
      echo "[saildrone_entrypoint] Virtual joystick page not found: ${uri}" >&2
      return
    fi
    uri="file://${uri}"
  fi

  (
    echo "[saildrone_entrypoint] Opening virtual joystick: ${uri}"
    sleep "${VJOY_HTML_DELAY}"
    open_uri "${uri}"
  ) >/tmp/vjoy_html.log 2>&1 &
}

wait_for_qgc() {
  if [ -n "${qgc_pid}" ]; then
    wait "${qgc_pid}"
    return $?
  fi

  sleep infinity
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
  local headless_launch_arg="true"

  if [ "${GZ_SERVER_ONLY}" = "0" ]; then
    headless_launch_arg="false"
  fi

  trap forward_signal INT TERM

  while true; do
    echo "[saildrone_entrypoint] Starting default launch (attempt ${attempt})"

    ros2 launch "${SAILDRONE_LAUNCH_PACKAGE}" "${SAILDRONE_LAUNCH_FILE}" \
      gui:="${SAILDRONE_GUI}" \
      use_sim_time:="${SAILDRONE_USE_SIM_TIME}" \
      paused:="${SAILDRONE_PAUSED}" \
      headless:="${headless_launch_arg}" \
      namespace:="${SAILDRONE_NAMESPACE}" \
      world_name:="${SAILDRONE_WORLD_NAME}" \
      x:="${SAILDRONE_X}" \
      y:="${SAILDRONE_Y}" \
      z:="${SAILDRONE_Z}" \
      roll:="${SAILDRONE_ROLL}" \
      pitch:="${SAILDRONE_PITCH}" \
      yaw:="${SAILDRONE_YAW}" \
      use_ned_frame:="${SAILDRONE_USE_NED_FRAME}" &
    launch_pid=$!

    set +e
    wait "${launch_pid}"
    exit_code=$?
    set -e

    launch_pid=""

    if [ "${stop_requested}" -eq 1 ]; then
      exit "${exit_code}"
    fi

    if [ "${SAILDRONE_RESTART_LAUNCH}" != "1" ]; then
      exit "${exit_code}"
    fi

    if [ "${SAILDRONE_MAX_RESTARTS}" -gt 0 ] && [ "${attempt}" -ge "${SAILDRONE_MAX_RESTARTS}" ]; then
      echo "[saildrone_entrypoint] Launch exited with code ${exit_code}; reached restart limit (${SAILDRONE_MAX_RESTARTS})" >&2
      exit "${exit_code}"
    fi

    echo "[saildrone_entrypoint] Launch exited with code ${exit_code}; restarting in ${SAILDRONE_RESTART_DELAY}s" >&2
    sleep "${SAILDRONE_RESTART_DELAY}"
    attempt=$((attempt + 1))
  done
}

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

case "${SAILDRONE_AUTOSTART}" in
  sim)
    start_qgc || true
    start_vjoy_html
    run_default_launch
    ;;
  qgc)
    START_QGC=1
    start_qgc
    wait_for_qgc
    ;;
  both)
    START_QGC=1
    start_qgc || true
    start_vjoy_html
    run_default_launch
    ;;
  none|shell)
    exec bash
    ;;
  *)
    echo "[saildrone_entrypoint] Invalid SAILDRONE_AUTOSTART=${SAILDRONE_AUTOSTART}; expected sim, qgc, both, or none" >&2
    exit 64
    ;;
esac
