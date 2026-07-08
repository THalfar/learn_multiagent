# GPU Enforcement & Buffer Hygiene on HP Shifts

- **id**: `gpu-enforcement-buffer-hygiene-on-hp-shifts`
- **status**: proposed (confidence 0.50)
- **source env**: -
- **tags**: cuda, buffer-management, hyperparameter-shift, stability

## When to use
Container has CUDA but VRAM=0, or learning_rate/architecture changes significantly mid-training

## Procedure
When learning_rate changes by >10x or train_freq/batch_size shifts significantly, unconditionally purge the replay buffer (.pkl) before resuming. Keep model weights and VecNormalize stats for continuity. Stale off-policy data guarantees critic collapse and entropy rebound. Only resume buffers when hyperparameters remain identical to the checkpoint.

## Pitfalls
Resuming old buffers with new LR causes immediate policy collapse; CPU fallback masks true training speed and wastes wall-clock time

## Verification
VRAM > 0GB, no timeout on 200k+ steps, reward curve stabilizes/increases post-resume without sudden drops
