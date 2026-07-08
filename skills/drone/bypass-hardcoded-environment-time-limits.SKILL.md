# Bypass Hardcoded Environment Time Limits

- **id**: `bypass-hardcoded-environment-time-limits`
- **status**: verified (confidence 0.80)
- **source env**: -
- **tags**: time-limit, env-override, wrapper-chain, gymnasium

## When to use
Episodes truncate at fixed steps despite gym.make(max_episode_steps=X) or registry overrides, indicating internal env enforcement

## Procedure
Wrap base env with a custom HorizonOverride class that tracks step_count in step(), forces truncated=False and info['TimeLimit.truncated']=False until desired limit, then yields control. Apply before observation/action wrappers.

## Pitfalls
Gymnasium's TimeLimit wrapper passes through inner env truncation flags; kwarg overrides fail against hardcoded __init__ limits

## Verification
Episode lengths exceed previous caps; info dict no longer shows early truncation; reward curve begins climbing past noise floor
