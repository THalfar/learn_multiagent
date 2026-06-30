# Solve panda / robotic manipulation

- **id**: `solve-panda-robotic-manipulation`
- **status**: verified (confidence 0.90)
- **source env**: PandaReach-v3
- **tags**: general

## When to use
An env like PandaReach-v3.

## Procedure
Use SAC with policy='MlpPolicy'; read max_episode_steps from env.spec and set learning_starts >= that value; each optimization iteration resume from the checkpoint (SAC.load + load_replay_buffer, learn one ~150k chunk, then save model + replay buffer).

## Pitfalls
Commit to one algorithm so checkpoint-resume accumulates.

## Verification
RESULT mean_reward >= the env threshold.
