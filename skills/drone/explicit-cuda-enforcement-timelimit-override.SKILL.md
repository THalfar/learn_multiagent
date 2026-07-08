# Explicit CUDA Enforcement & TimeLimit Override

- **id**: `explicit-cuda-enforcement-timelimit-override`
- **status**: proposed (confidence 0.50)
- **source env**: -
- **tags**: cuda, time-limit, env-config, performance

## When to use
Container has CUDA but VRAM=0, or env.spec.max_episode_steps modification fails post-gym.make

## Procedure
Force device='cuda' if torch.cuda.is_available(). Wrap base env with gym.wrappers.TimeLimit(env, max_episode_steps=1000) immediately after gym.make() to bypass baked-in TimeLimit wrappers.

## Pitfalls
device='auto' silently falls back to CPU in some SB3 versions; modifying spec.post-make is ignored by Gymnasium's wrapper chain

## Verification
VRAM > 0GB, episodes exceed previous step caps, training speed increases significantly
