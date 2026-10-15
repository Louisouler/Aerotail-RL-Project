#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# demo.sh - one-command start/stop of the live demo environment:
#   Xvfb :99  ->  AirSim Blocks (windowed on :99)  ->  dashboard (127.0.0.1:8050)
#
#   dashboard/demo.sh start          start whatever is not running yet
#   dashboard/demo.sh stop           stop the dashboard
#   dashboard/demo.sh stop --all     also stop the simulator and Xvfb
#                                    (refused while a training job is running)
#   dashboard/demo.sh restart        restart the dashboard
#   dashboard/demo.sh status
#
# Processes are found by exact process name or by the listening port - never
# with `pkill -f <pattern>`, which can match unrelated command lines.
# Logs: dashboard/runtime/{xvfb,blocks,dashboard}.log
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT_DIR=/mnt/disks/data/aerotail
DASH_DIR="$PROJECT_DIR/dashboard"
RUN_DIR="$DASH_DIR/runtime"
BLOCKS_DIR="$PROJECT_DIR/sim/Blocks/Blocks/LinuxNoEditor"
DASH_PY=/mnt/disks/data/aerotail-dashboard-venv/bin/python
EXP_PY=/mnt/disks/data/aerotail-venv/bin/python
DISPLAY_NUM=99
PORT=8050
AIRSIM_PORT=41451

mkdir -p "$RUN_DIR"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
xvfb_pid() { pgrep -x Xvfb -a | awk -v d=":$DISPLAY_NUM" '$3 == d {print $1; exit}'; }
blocks_pid() { pgrep -x Blocks | head -1; }
dash_pid() {  # the process listening on $PORT, if it is the dashboard
    local pid
    pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$pid" ] && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "app.py" && echo "$pid"
}
training_running() {
    "$DASH_PY" -c "import sys; sys.path.insert(0, '$DASH_DIR'); import control; sys.exit(0 if control.running_training() else 1)"
}

wait_for() {  # wait_for <seconds> <description> <command...>
    local secs=$1 what=$2; shift 2
    for _ in $(seq 1 "$secs"); do "$@" && return 0; sleep 1; done
    echo "  ! $what did not come up within ${secs}s"; return 1
}

start() {
    if [ -z "$(xvfb_pid)" ]; then
        echo "- starting Xvfb :$DISPLAY_NUM"
        setsid nohup Xvfb ":$DISPLAY_NUM" -screen 0 1280x720x24 < /dev/null > "$RUN_DIR/xvfb.log" 2>&1 &
        wait_for 10 "Xvfb" test -S "/tmp/.X11-unix/X$DISPLAY_NUM" || return 1
    else
        echo "- Xvfb :$DISPLAY_NUM already running (pid $(xvfb_pid))"
    fi

    if [ -z "$(blocks_pid)" ]; then
        echo "- starting AirSim Blocks on :$DISPLAY_NUM (about 30 s)"
        # exec: no wrapper shell is left holding this script's stdout
        (cd "$BLOCKS_DIR" && DISPLAY=":$DISPLAY_NUM" exec setsid nohup ./Blocks.sh -windowed -ResX=1280 -ResY=720 \
            < /dev/null > "$RUN_DIR/blocks.log" 2>&1) &
        wait_for 90 "AirSim RPC port $AIRSIM_PORT" port_open "$AIRSIM_PORT" || return 1
        "$EXP_PY" -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('  vehicles:', c.listVehicles())" \
            2>/dev/null | tail -1
    else
        echo "- AirSim Blocks already running (pid $(blocks_pid))"
    fi

    if [ -z "$(dash_pid)" ]; then
        if port_open "$PORT"; then
            echo "  ! port $PORT is already in use by another process - not starting the dashboard"; return 1
        fi
        echo "- starting dashboard on 127.0.0.1:$PORT"
        (cd "$DASH_DIR" && exec setsid nohup "$DASH_PY" app.py --port "$PORT" < /dev/null > "$RUN_DIR/dashboard.log" 2>&1) &
        wait_for 20 "dashboard" port_open "$PORT" || return 1
    else
        echo "- dashboard already running (pid $(dash_pid))"
    fi

    local ip
    ip=$(curl -s -m 2 -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" \
        || echo "<vm-ip>")
    echo
    echo "Ready. On your own computer run:"
    echo "  ssh -i ~/.ssh/gcp_key -L $PORT:127.0.0.1:$PORT $(whoami)@$ip"
    echo "then open http://localhost:$PORT"
}

stop_dashboard() {
    local pid; pid=$(dash_pid)
    if [ -n "$pid" ]; then
        kill "$pid" && echo "- dashboard stopped (pid $pid)"
        wait_for 10 "dashboard shutdown" bash -c "! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null"
    else
        echo "- dashboard not running"
    fi
}

stop_all() {
    stop_dashboard
    if training_running; then
        echo "  ! a training job is running - leaving the simulator and Xvfb up"; return 1
    fi
    local pid
    pid=$(blocks_pid); [ -n "$pid" ] && kill "$pid" && echo "- AirSim Blocks stopped (pid $pid)"
    pid=$(xvfb_pid);   [ -n "$pid" ] && kill "$pid" && echo "- Xvfb :$DISPLAY_NUM stopped (pid $pid)"
    return 0
}

state() { [ -n "$1" ] && echo "running (pid $1)${2:-}" || echo "stopped"; }

status() {
    echo "Xvfb :$DISPLAY_NUM      $(state "$(xvfb_pid)")"
    echo "AirSim Blocks  $(state "$(blocks_pid)" ", RPC port $AIRSIM_PORT $(port_open $AIRSIM_PORT && echo open || echo closed)")"
    echo "dashboard      $(state "$(dash_pid)" ", http://127.0.0.1:$PORT")"
    "$DASH_PY" -c "
import sys; sys.path.insert(0, '$DASH_DIR'); import control
d = control.DemoController().status(); t = control.running_training()
print('demo run      ', f\"running (pid {d['pid']})\" if d['running'] else 'idle')
print('training      ', f'RUNNING (pid {t[0][0]})' if t else 'none')"
}

case "${1:-}" in
    start)   start ;;
    stop)    if [ "${2:-}" = "--all" ]; then stop_all; else stop_dashboard; fi ;;
    restart) stop_dashboard; sleep 1; start ;;
    status)  status ;;
    *)       echo "usage: $0 {start|stop [--all]|restart|status}"; exit 2 ;;
esac
