#!/usr/bin/env bash
# Starts RecurrentPPO stage 1 automatically once the stage-0 retry run exits.
#
# Usage: chain_stage1.sh <pid of the stage-0 09_train_with_retry.sh process>
#
# Stage 1 is only launched if stage 0 actually finished: final_model.zip and
# vecnormalize.pkl must both exist in the stage-0 checkpoint dir. The stage-1
# timestep budget is read from $STAGE1_STEPS_FILE at launch time, so it can be
# changed while this script is waiting. Actions are logged to $CHAIN_LOG.
set -u

STAGE0_PID="$1"
PROJECT_DIR=/mnt/disks/data/aerotail
SCRATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE1_STEPS_FILE="$SCRATCH_DIR/stage1_timesteps.txt"
CHAIN_LOG="$PROJECT_DIR/logs/train_chain.log"
STAGE0_DIR="$PROJECT_DIR/checkpoints/rl/active/recurrent_ppo_stage0"
STAGE1_LR=3e-5   # same stage-1 fine-tuning LR as the PPO/SAC stage-1 models

log() { echo "$(date -u '+%F %T UTC') $*" >> "$CHAIN_LOG"; }

log "chain started: waiting for stage-0 retry run (pid $STAGE0_PID) to exit"
while kill -0 "$STAGE0_PID" 2>/dev/null; do
    sleep 60
done
log "stage-0 retry run exited"

if [ ! -f "$STAGE0_DIR/final_model.zip" ] || [ ! -f "$STAGE0_DIR/vecnormalize.pkl" ]; then
    log "ERROR: stage 0 did not produce final_model.zip + vecnormalize.pkl - NOT starting stage 1"
    exit 1
fi

STAGE1_STEPS=$(tr -dc '0-9' < "$STAGE1_STEPS_FILE")
if [ -z "$STAGE1_STEPS" ] || [ "$STAGE1_STEPS" -le 0 ]; then
    log "ERROR: invalid stage-1 timestep budget in $STAGE1_STEPS_FILE - NOT starting stage 1"
    exit 1
fi

log "starting stage 1: $STAGE1_STEPS timesteps, lr $STAGE1_LR, bootstrapped from $STAGE0_DIR/final_model.zip"
cd "$PROJECT_DIR"
PYTHONUNBUFFERED=1 setsid nohup bash scripts/09_train_with_retry.sh recurrent_ppo 1 "$STAGE1_STEPS" Drone1 Target1 \
    "$STAGE1_LR" "$STAGE0_DIR/final_model.zip" "$STAGE0_DIR/vecnormalize.pkl" \
    > "$PROJECT_DIR/logs/train_recurrent_ppo_stage1.out" 2>&1 < /dev/null &
log "stage-1 retry run launched (pid $!)"
