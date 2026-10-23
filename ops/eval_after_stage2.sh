#!/usr/bin/env bash
# Waits for the RecurrentPPO stage-2 retry run to exit, then evaluates the
# stage-2 model with ops/eval_single.py (07 protocol; other rows untouched).
# Progress is logged to logs/eval_stage2_runner.log; the evaluation output goes
# to logs/eval_recurrent_ppo_stage2.out.
set -u

PROJECT_DIR=/mnt/disks/data/aerotail
STAGE2_DIR="$PROJECT_DIR/checkpoints/rl/active/recurrent_ppo_stage2"
RUNNER_LOG="$PROJECT_DIR/logs/eval_stage2_runner.log"

log() { echo "$(date -u '+%F %T UTC') $*" >> "$RUNNER_LOG"; }

log "waiting for the stage-2 retry run to exit"
while ps -eo args --no-headers | grep -qE '^bash scripts/09_train_with_retry.sh recurrent_ppo 2'; do
    sleep 30
done
log "stage-2 retry run exited"

watchdog=$(ps -eo pid=,args= | awk '$2 == "bash" && $3 == "ops/train_watchdog.sh" {print $1}')
if [ -n "$watchdog" ]; then
    kill $watchdog && log "watchdog stopped (pid $watchdog)"
    echo "$(date -u '+%F %T UTC') watchdog stopped after stage-2 training" >> "$PROJECT_DIR/logs/train_watchdog.log"
fi

if [ ! -f "$STAGE2_DIR/final_model.zip" ] || [ ! -f "$STAGE2_DIR/vecnormalize.pkl" ]; then
    log "ERROR: stage 2 did not produce final_model.zip + vecnormalize.pkl - not evaluating"
    exit 1
fi

if ! (exec 3<>/dev/tcp/127.0.0.1/41451) 2>/dev/null; then
    log "simulator not reachable - starting it with dashboard/demo.sh"
    "$PROJECT_DIR/dashboard/demo.sh" start >> "$RUNNER_LOG" 2>&1
fi

cd "$PROJECT_DIR"
source /mnt/disks/data/aerotail-venv/bin/activate
log "evaluating recurrent_ppo stage 2"
PYTHONUNBUFFERED=1 python3 ops/eval_single.py recurrent_ppo 2 > "$PROJECT_DIR/logs/eval_recurrent_ppo_stage2.out" 2>&1
log "eval_single.py exited with code $?"
