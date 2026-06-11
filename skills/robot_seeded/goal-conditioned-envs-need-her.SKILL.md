# Goal-conditioned envs need HER

- **id**: `goal-conditioned-envs-need-her`
- **status**: verified (confidence 0.50)
- **source env**: -
- **tags**: goal, her, manipulation, robotics, success_rate

## When to use
Dict observation with achieved_goal/desired_goal (panda/fetch robotic manipulation), sparse reward.

## Procedure
Use SAC with policy='MultiInputPolicy', replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs={'n_sampled_goal':4,'goal_selection_strategy':'future'}. Read max_episode_steps from env.spec and set learning_starts >= that. Each optimization iteration resume from the checkpoint: SAC.load(path, env=env) + model.load_replay_buffer(buf), then print(f'RESUMED: buffer_transitions={model.replay_buffer.size()}'), learn one wall-clock-sized chunk, then save model + replay buffer. Report success_rate (the is_success fraction over >= 20 eval episodes with fixed seeds), not the raw sparse reward.

## Pitfalls
Plain SAC/PPO/TD3 without HER cannot solve PandaPush/PandaPickAndPlace. Never pass reset_num_timesteps=False (breaks termination on a reloaded model). 30k steps is starvation - size the chunk from measured steps/s and let chunks accumulate. Resuming the model WITHOUT load_replay_buffer silently forgets everything - the Tester rejects a resume without the RESUMED proof line.

## Verification
RESULT line prints success_rate in [0,1] >= the env success_threshold.
