#!/usr/bin/env bash
# Training watchdog for scripts/09_train_with_retry.sh runs.
#
# Problem it solves: when the simulator crashes, the trainer (04_train_rl.py)
# does not exit - the AirSim RPC client blocks for its default 3600 s timeout
# before raising, so the retry loop only restarts ~1 hour later. This watchdog
# terminates a trainer whose simulator is gone (or whose log has stalled), so
# 09_train_with_retry.sh retries immediately from the latest checkpoint.
#
# It never touches the simulator, checkpoints or code - it only signals the
# trainer process. Actions are appended to $WATCHDOG_LOG.
set -u

PROJECT_DIR=/mnt/disks/data/aerotail
WATCHDOG_LOG="$PROJECT_DIR/logs/train_watchdog.log"
CHECK_INTERVAL=30          # seconds between checks
SIM_MISSING_LIMIT=2        # consecutive checks without simulator -> act (~60 s)
STALL_LIMIT=900            # seconds without training-log output -> act

log() { echo "$(date -u '+%F %T UTC') $*" >> "$WATCHDOG_LOG"; }

kill_trainer() {
    local pids="$1" reason="$2"
    log "ACTION: terminating trainer pid(s) $pids - $reason"
    kill -TERM $pids 2>/dev/null
    sleep 10
    for p in $pids; do
        if kill -0 "$p" 2>/dev/null; then
            log "trainer pid $p still alive after TERM - sending KILL"
            kill -KILL "$p" 2>/dev/null
        fi
    done
}

log "watchdog started (interval ${CHECK_INTERVAL}s, sim-missing limit ${SIM_MISSING_LIMIT}, stall limit ${STALL_LIMIT}s)"
sim_missing=0
while true; do
    # Anchored match: only the trainer launched by 09 ("python3 scripts/04_train_rl.py ...");
    # an unanchored -f match would also hit any shell whose command line mentions the script.
    trainer_pids=$(pgrep -f "^python3 scripts/04_train_rl\.py " | tr '\n' ' ')
    if [ -n "${trainer_pids// /}" ]; then
        # Simulator binary has the exact process name of the Unreal executable.
        if pgrep -x "Blo""cks" >/dev/null; then
            sim_missing=0
        else
            sim_missing=$((sim_missing + 1))
            log "simulator not running while trainer is alive (check $sim_missing/$SIM_MISSING_LIMIT)"
        fi
        if [ "$sim_missing" -ge "$SIM_MISSING_LIMIT" ]; then
            kill_trainer "$trainer_pids" "simulator process is gone (crash)"
            sim_missing=0
        else
            train_log=$(ls -t "$PROJECT_DIR"/logs/train_*.out 2>/dev/null | head -1)
            if [ -n "$train_log" ]; then
                age=$(( $(date +%s) - $(stat -c %Y "$train_log") ))
                if [ "$age" -ge "$STALL_LIMIT" ]; then
                    kill_trainer "$trainer_pids" "no training-log output for ${age}s ($train_log)"
                fi
            fi
        fi
    else
        sim_missing=0
    fi
    sleep "$CHECK_INTERVAL"
done
