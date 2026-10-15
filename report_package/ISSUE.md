## AirSim reset/takeoff issue

### Symptom

After some episode resets, the drone starts at or near ground level instead of taking off, causing early ground collisions.

### Root cause

`simSetVehiclePose()` teleports the drone to the spawn pose but preserves its previous linear velocity.

If the drone is still moving faster than approximately 0.05 m/s, AirSim rejects `takeoffAsync()` with:

`Cannot perform takeoff because vehicle is already moving`

The RPC server suppresses this exception, so `env.reset()` appears to complete successfully even though takeoff failed.

### Impact

- Early ground collisions after reset

- Inflated collision rate

- Reduced episode length

- Reduced total reward

- Unfair comparison against the authoritative SAC/PPO evaluation results

### Evidence

The issue was reproduced deterministically:

- stationary drone -&gt; takeoff succeeds

- moving drone -&gt; takeoff fails

- waiting until linear speed &lt;= 0.05 m/s -&gt; takeoff succeeds

The issue is not caused by the selected AirSim simulator version; the relevant takeoff logic is unchanged between the tested candidate versions.

### Proposed fix

Before `takeoffAsync()`, wait until the drone's linear speed is &lt;= 0.05 m/s, with a bounded timeout.

This preserves:

- observation semantics

- reward function

- curriculum

- spawn pose

- collision logic

- action semantics

### Validation required

After applying the fix:

1. rerun smoke test

2. rerun PPO/SAC Stage 0–1 fidelity evaluation

3. confirm early ground-start collisions disappear

4. confirm metrics return close to the authoritative baseline