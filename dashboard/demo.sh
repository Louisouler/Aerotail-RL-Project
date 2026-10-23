#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# demo.sh - one-command start/stop of the live demo environment:
#   Xvfb :99  ->  AirSim environment (windowed on :99)  ->  dashboard (127.0.0.1:8050)
#
#   dashboard/demo.sh start [--env E] start whatever is not running yet;
#                                    E = blocks (default) or nh (AirSimNH neighbourhood)
#   dashboard/demo.sh stop           stop the dashboard
#   dashboard/demo.sh stop --all     also stop the simulator and Xvfb
#                                    (refused while a training job is running)
#   dashboard/demo.sh restart        restart the dashboard
#   dashboard/demo.sh status
#
# Processes are found by exact process name or by the listening port - never
# with `pkill -f <pattern>`, which can match unrelated command lines.
# Only one environment can run at a time (both use RPC port 41451): to switch,
# `stop --all`, then `start --env <other>`. Training always uses Blocks.
# Logs: dashboard/runtime/{xvfb,blocks,airsimnh,dashboard}.log
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT_DIR=/mnt/disks/data/aerotail
DASH_DIR="$PROJECT_DIR/dashboard"
RUN_DIR="$DASH_DIR/runtime"
# environment -> "<process name> <directory of its launch script>"
declare -A ENVS=(
    [blocks]="Blocks $PROJECT_DIR/sim/Blocks/Blocks/LinuxNoEditor"
    [nh]="AirSimNH $PROJECT_DIR/sim/AirSimNH/AirSimNH/LinuxNoEditor"
)
DASH_PY=/mnt/disks/data/aerotail-dashboard-venv/bin/python
EXP_PY=/mnt/disks/data/aerotail-venv/bin/python
DISPLAY_NUM=99
PORT=8050
AIRSIM_PORT=41451

mkdir -p "$RUN_DIR"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
xvfb_pid() { pgrep -x Xvfb -a | awk -v d=":$DISPLAY_NUM" '$3 == d {print $1; exit}'; }
sim_running() {  # prints "<process name> <pid>" of the running environment, if any
    local e name pid
    for e in "${!ENVS[@]}"; do
        name=${ENVS[$e]%% *}
        pid=$(pgrep -x "$name" | head -1)
        [ -n "$pid" ] && { echo "$name $pid"; return; }
    done
}
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
    local env=$1 name dir running
    if [ -z "${ENVS[$env]:-}" ]; then
        echo "unknown environment '$env' (choose: ${!ENVS[*]})"; return 2
    fi
    name=${ENVS[$env]%% *}; dir=${ENVS[$env]#* }
    if [ ! -x "$dir/$name.sh" ]; then
        echo "  ! $dir/$name.sh not found - is the environment downloaded?"; return 1
    fi

    if [ -z "$(xvfb_pid)" ]; then
        echo "- starting Xvfb :$DISPLAY_NUM"
        setsid nohup Xvfb ":$DISPLAY_NUM" -screen 0 1280x720x24 < /dev/null > "$RUN_DIR/xvfb.log" 2>&1 &
        wait_for 10 "Xvfb" test -S "/tmp/.X11-unix/X$DISPLAY_NUM" || return 1
    else
        echo "- Xvfb :$DISPLAY_NUM already running (pid $(xvfb_pid))"
    fi

    running=$(sim_running)
    if [ -z "$running" ]; then
        echo "- starting AirSim $name on :$DISPLAY_NUM (about 30 s)"
        # exec: no wrapper shell is left holding this script's stdout
        (cd "$dir" && DISPLAY=":$DISPLAY_NUM" exec setsid nohup "./$name.sh" -windowed -ResX=1280 -ResY=720 \
            < /dev/null > "$RUN_DIR/${name,,}.log" 2>&1) &
        wait_for 90 "AirSim RPC port $AIRSIM_PORT" port_open "$AIRSIM_PORT" || return 1
        "$EXP_PY" -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('  vehicles:', c.listVehicles())" \
            2>/dev/null | tail -1
    elif [ "${running% *}" = "$name" ]; then
        echo "- AirSim $name already running (pid ${running#* })"
    else
        echo "  ! AirSim ${running% *} is running (pid ${running#* }) - run '$0 stop --all' first to switch to $name"
        return 1
    fi

    start_dashboard
}

start_dashboard() {
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
    local pid running
    running=$(sim_running)
    [ -n "$running" ] && kill "${running#* }" && echo "- AirSim ${running% *} stopped (pid ${running#* })"
    pid=$(xvfb_pid);   [ -n "$pid" ] && kill "$pid" && echo "- Xvfb :$DISPLAY_NUM stopped (pid $pid)"
    return 0
}

state() { [ -n "$1" ] && echo "running (pid $1)${2:-}" || echo "stopped"; }

status() {
    echo "Xvfb :$DISPLAY_NUM      $(state "$(xvfb_pid)")"
    local running; running=$(sim_running)
    echo "AirSim         ${running:+${running% *} }$(state "${running#* }" \
        ", RPC port $AIRSIM_PORT $(port_open $AIRSIM_PORT && echo open || echo closed)")"
    echo "dashboard      $(state "$(dash_pid)" ", http://127.0.0.1:$PORT")"
    "$DASH_PY" -c "
import sys; sys.path.insert(0, '$DASH_DIR'); import control
d = control.DemoController().status(); t = control.running_training()
print('demo run      ', f\"running (pid {d['pid']})\" if d['running'] else 'idle')
print('training      ', f'RUNNING (pid {t[0][0]})' if t else 'none')"
}

case "${1:-}" in
    start)   if [ "${2:-}" = "--env" ]; then start "${3:-}"; else start blocks; fi ;;
    stop)    if [ "${2:-}" = "--all" ]; then stop_all; else stop_dashboard; fi ;;
    restart) stop_dashboard; sleep 1; start_dashboard ;;
    status)  status ;;
    *)       echo "usage: $0 {start [--env blocks|nh]|stop [--all]|restart|status}"; exit 2 ;;
esac
