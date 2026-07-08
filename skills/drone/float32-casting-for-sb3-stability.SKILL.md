# Float32 Casting for SB3 Stability

- **id**: `float32-casting-for-sb3-stability`
- **status**: proposed (confidence 0.50)
- **source env**: -
- **tags**: dtype, stability, sb3, float64

## When to use
Gymnasium envs output float64 observations, causing PyTorch autograd instability in SB3

## Procedure
VecNormalize.load() promotes observations to float64 because its internal stats are float64. CRITICAL: Immediately after loading, cast the stats: vec_env.ob_mean = vec_env.ob_mean.astype(np.float32); vec_env.ob_std = vec_env.ob_std.astype(np.float32). This stops promotion at the source. Keep explicit obs casting before predict as a safety net.

## Pitfalls
Skipping this causes NaN gradients, policy collapse, and deterministic crash loops regardless of timesteps or architecture

## Verification
No float64 warnings; training and eval both proceed without gradient explosions or reward drops; episode lengths vary instead of fixing at minimum steps
