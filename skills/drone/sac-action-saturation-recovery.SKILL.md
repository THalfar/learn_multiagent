# SAC Action Saturation Recovery

- **id**: `sac-action-saturation-recovery`
- **status**: proposed (confidence 0.50)
- **source env**: -
- **tags**: sac, saturation, continuous-control, hyperparameter-tuning

## When to use
Continuous control envs where action_saturation > 0.8 and episodes terminate prematurely with max-thrust/crash behavior

## Procedure
Reduce learning_rate to 1e-4, ensure ent_coef='auto'. Monitor entropy; if policy saturates, the critic is overconfident in boundary actions. Stabilize updates before scaling timesteps or widening networks.

## Pitfalls
Increasing timesteps or architecture while saturated only cements the crash policy; always fix LR first

## Verification
action_saturation drops below 0.5, episode lengths increase beyond minimum steps, entropy stabilizes near target_entropy
