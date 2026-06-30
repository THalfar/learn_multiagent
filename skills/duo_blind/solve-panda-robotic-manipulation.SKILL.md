# Solve panda / robotic manipulation

- **id**: `solve-panda-robotic-manipulation`
- **status**: verified (confidence 1.00)
- **source env**: PandaReach-v3
- **tags**: goal, her, manipulation, robotics, success_rate

## When to use
A goal-conditioned / sparse-reward env like PandaReach-v3 (Dict obs with desired_goal).

## Procedure
Use SAC with policy='MlpPolicy'; read max_episode_steps from env.spec and set learning_starts >= that value; each optimization iteration resume from the checkpoint (SAC.load + load_replay_buffer, learn one wall-clock-sized chunk computed from measured steps/s, then save model + replay buffer); report success_rate (the is_success fraction), never the raw sparse reward.

## Pitfalls
never pass reset_num_timesteps=False (breaks termination on a reloaded model); starving the chunk stalls progress - size it from measured steps/s and accumulate.

## Verification
RESULT line prints success_rate in [0,1] >= the env threshold.
