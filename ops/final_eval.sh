#!/usr/bin/env bash
# Waits for the RecurrentPPO stage-1 retry run to finish cleanly, then runs
# scripts/07_evaluate_all.py once over every configured model.
# Progress is logged to logs/final_eval_runner.log; the evaluation output goes
# to logs/eval_all_final.out.
set -u

PROJECT_DIR=/mnt/disks/data/aerotail
STAGE1_DIR="$PROJECT_DIR/checkpoints/rl/active/recurrent_ppo_stage1"
RUNNER_LOG="$PROJECT_DIR/logs/final_eval_runner.log"
BLOCKS_DIR="$PROJECT_DIR/sim/Blocks/Blocks/LinuxNoEditor"

log() { echo "$(date -u '+%F %T UTC') $*" >> "$RUNNER_LOG"; }

log "waiting for the stage-1 retry run to exit"
while ps -eo args --no-headers | grep -qE '^bash scripts/09_train_with_retry.sh recurrent_ppo 1'; do
    sleep 30
done
log "stage-1 retry run exited"

if [ ! -f "$STAGE1_DIR/final_model.zip" ] || [ ! -f "$STAGE1_DIR/vecnormalize.pkl" ]; then
    log "ERROR: stage 1 did not produce final_model.zip + vecnormalize.pkl - not evaluating"
    exit 1
fi

if ! ps -o comm= -C Blocks >/dev/null; then
    log "simulator not running - launching it"
    cd "$BLOCKS_DIR"
    xvfb-run -a -s "-screen 0 1024x768x24" ./Blocks.sh -RenderOffscreen -windowed -ResX=1024 -ResY=768 > /tmp/blocks.log 2>&1 &
    sleep 25
fi

cd "$PROJECT_DIR"
source /mnt/disks/data/aerotail-venv/bin/activate
log "running scripts/07_evaluate_all.py"
PYTHONUNBUFFERED=1 python3 scripts/07_evaluate_all.py > "$PROJECT_DIR/logs/eval_all_final.out" 2>&1
log "07_evaluate_all.py exited with code $?"
