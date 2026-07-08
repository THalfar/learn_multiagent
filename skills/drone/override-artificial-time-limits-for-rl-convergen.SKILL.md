# Override Artificial Time Limits for RL Convergence

- **id**: `override-artificial-time-limits-for-rl-convergen`
- **status**: proposed (confidence 0.50)
- **source env**: -
- **tags**: time-limit, env-config, rl-convergence

## When to use
Episodes terminate prematurely due to env.spec.max_episode_steps being too short for policy learning (e.g., 4 steps)

## Procedure
Gymnasium auto-wraps with TimeLimit using registry specs. To override: unwrap first then re-wrap (env = gym.wrappers.TimeLimit(gym.make('hover-aviary-v0').unwrapped, max_episode_steps=1000)), or pass max_episode_steps=1000 directly to gym.make(). Nesting TimeLimits causes the inner default to trigger first.

## Pitfalls
Training on artificially short episodes causes reward collapse and prevents credit assignment; never accept default time limits blindly

## Verification
Episode lengths exceed previous caps; info dict no longer shows TimeLimit.truncated at step 4; reward curve begins to climb
