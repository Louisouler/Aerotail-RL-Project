#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 09_train_with_retry.sh
#
# We've now hit unexplained AirSim RPC crashes/timeouts several times during
# long unattended runs (once from a SIGHUP on the sim process, once from a
# plain request timeout, neither tied to the concurrent-sim-access mistake
# we separately identified and fixed). Rather than keep manually noticing,
# diagnosing, and restarting by hand, we wrap training in a retry loop: on
# any crash, we kill and relaunch the simulator fresh and resume training
# from the latest saved checkpoint (now saved every 5000 steps, not 20000,
# so a crash loses at most a few minutes of progress instead of potentially
# an entire run).
#
# Usage: ./scripts/09_train_with_retry.sh <algo> <stage> <target_timesteps>
# Example: ./scripts/09_train_with_retry.sh ppo 0 500000
# ---------------------------------------------------------------------------
set -uo pipefail  # NOT -e: we want to catch failures ourselves, not exit on them

ALGO="$1"
STAGE="$2"
TARGET_TIMESTEPS="$3"
VEHICLE="${4:-Drone1}"
TARGET_VEHICLE="${5:-Target1}"
LR_OVERRIDE="${6:-}"
# Used ONLY to bootstrap a brand-new stage (e.g. stage2) from the previous
# stage's finished checkpoint, on the very first attempt when this stage's
# own checkpoint dir is still empty. Ignored on any retry once this stage
# has its own checkpoints to resume from instead.
INITIAL_CHECKPOINT="${7:-}"
# For PPO/RecurrentPPO: the VecNormalize stats file to seed this stage
# with, so normalization continues smoothly from the previous stage
# instead of resetting cold (we found resetting it destabilizes training).
VECNORM_INITIAL="${8:-}"
# Used ONLY to bootstrap a brand-new stage (e.g. stage2) from the previous
# stage's finished checkpoint, on the very first attempt when this stage's
# own checkpoint dir is still empty. Ignored on any retry once this stage
# has its own checkpoints to resume from instead.
INITIAL_CHECKPOINT="${7:-}"
# For PPO/RecurrentPPO: the VecNormalize stats file to seed this stage
# with, so normalization continues smoothly from the previous stage
# instead of resetting cold (we found resetting it destabilizes training).
VECNORM_INITIAL="${8:-}"

DATA_DIR=/mnt/disks/data
PROJECT_DIR="$DATA_DIR/aerotail"
CKPT_DIR="$PROJECT_DIR/checkpoints/rl/active/${ALGO}_stage${STAGE}"
BLOCKS_DIR="$PROJECT_DIR/sim/Blocks/Blocks/LinuxNoEditor"
MAX_RETRIES=15

source "$DATA_DIR/aerotail-venv/bin/activate"
mkdir -p "$CKPT_DIR"

launch_sim() {
    echo "== Killing any existing sim processes =="
    pkill -9 -f Blocks 2>/dev/null
    pkill -9 -f Xvfb 2>/dev/null
    sleep 3
    echo "== Launching fresh AirSim instance =="
    cd "$BLOCKS_DIR"
    xvfb-run -a -s "-screen 0 1024x768x24" ./Blocks.sh -RenderOffscreen -windowed -ResX=1024 -ResY=768 > /tmp/blocks.log 2>&1 &
    sleep 25
    python3 -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('sim ready:', c.listVehicles())"
    cd "$PROJECT_DIR"
}

# Find the most recent checkpoint (highest step count) for this algo/stage,
# if any exists, and how many timesteps it represents - SB3's
# CheckpointCallback names files like ckpt_<algo>_<N>_steps.zip.
find_latest_checkpoint() {
    ls -1 "$CKPT_DIR"/ckpt_*_steps.zip 2>/dev/null | \
        sed -E 's/.*_([0-9]+)_steps\.zip/\1 &/' | sort -n | tail -1
}

# Checkpoints are named by the model's CUMULATIVE timestep count, which
# continues from the previous stage when a stage is bootstrapped from
# INITIAL_CHECKPOINT. Record where this stage started so progress is measured
# within the stage - otherwise the first crash in e.g. stage 1 looked like
# "target already reached" and the run silently stopped.
STAGE_START_FILE="$CKPT_DIR/stage_start_timesteps.txt"
if [ ! -f "$STAGE_START_FILE" ]; then
    if [ -n "$INITIAL_CHECKPOINT" ]; then
        python3 -c "import json, sys, zipfile; print(json.loads(zipfile.ZipFile(sys.argv[1]).read('data'))['num_timesteps'])" \
            "$INITIAL_CHECKPOINT" > "$STAGE_START_FILE" || rm -f "$STAGE_START_FILE"
    else
        echo 0 > "$STAGE_START_FILE"
    fi
fi
STAGE_START=$(cat "$STAGE_START_FILE" 2>/dev/null || echo 0)
echo "Stage starts at cumulative timestep $STAGE_START"

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "########## Attempt $attempt / $MAX_RETRIES ##########"

    latest_line=$(find_latest_checkpoint)
    if [ -n "$latest_line" ]; then
        done_steps=$(( $(echo "$latest_line" | awk '{print $1}') - STAGE_START ))
        ckpt_path=$(echo "$latest_line" | awk '{print $2}')
    else
        done_steps=0
        ckpt_path=""
    fi

    remaining=$((TARGET_TIMESTEPS - done_steps))
    echo "Progress so far: $done_steps / $TARGET_TIMESTEPS timesteps"

    if [ "$remaining" -le 0 ]; then
        echo "Target already reached. Done."
        break
    fi

    launch_sim
    if [ $? -ne 0 ]; then
        echo "Sim failed to come up, retrying..."
        continue
    fi

    if [ -z "$ckpt_path" ] && [ -n "$INITIAL_CHECKPOINT" ]; then
        echo "No checkpoint in this stage yet - bootstrapping from $INITIAL_CHECKPOINT"
        ckpt_path="$INITIAL_CHECKPOINT"
    fi

    if [ -z "$ckpt_path" ] && [ -n "$INITIAL_CHECKPOINT" ]; then
        echo "No checkpoint in this stage yet - bootstrapping from $INITIAL_CHECKPOINT"
        ckpt_path="$INITIAL_CHECKPOINT"
    fi

    if [ -n "$ckpt_path" ]; then
        echo "Resuming from $ckpt_path for $remaining more timesteps"
        LR_ARGS=""
        if [ -n "$LR_OVERRIDE" ]; then
            LR_ARGS="--learning-rate $LR_OVERRIDE"
        fi
        VECNORM_ARGS=""
        # Prefer this stage's own saved vecnormalize.pkl (if we're resuming
        # our own progress); otherwise fall back to the bootstrap one from
        # the previous stage, on the very first attempt only.
        own_vecnorm="$CKPT_DIR/vecnormalize.pkl"
        if [ -f "$own_vecnorm" ]; then
            VECNORM_ARGS="--vecnormalize-from $own_vecnorm"
        elif [ -n "$VECNORM_INITIAL" ]; then
            VECNORM_ARGS="--vecnormalize-from $VECNORM_INITIAL"
        fi
        python3 scripts/04_train_rl.py --algo "$ALGO" --stage "$STAGE" \
            --timesteps "$remaining" --vehicle "$VEHICLE" --target "$TARGET_VEHICLE" --load-from "$ckpt_path" $LR_ARGS $VECNORM_ARGS
    else
        echo "No checkpoint found, starting fresh for $remaining timesteps"
        python3 scripts/04_train_rl.py --algo "$ALGO" --stage "$STAGE" \
            --timesteps "$remaining" --vehicle "$VEHICLE" --target "$TARGET_VEHICLE"
    fi

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Training finished cleanly."
        break
    else
        echo "Training crashed (exit code $exit_code). Will retry after checking progress."
        sleep 5
    fi
done

if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "Hit max retries ($MAX_RETRIES) without reaching target - stopping. Check logs."
fi
