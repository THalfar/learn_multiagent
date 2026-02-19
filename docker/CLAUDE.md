# docker/ - RL Training Sandbox

## Dockerfile — GPU-accelerated Python sandbox
Base image: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`

### Installed stack
- Python 3 + pip
- PyTorch 2.7 (CUDA 12.8)
- `stable-baselines3[extra]` (includes tensorboard, tqdm, rich)
- gymnasium + gymnasium[classic-control,box2d,mujoco]
- scipy, pandas, seaborn, optuna, opencv-python-headless
- MuJoCo (system-level install)

### Security
- Non-root user (`rluser`)
- Workspace: `/workspace`
- Network disabled at runtime (`--network none` in docker run)
- Code mounted read-only

### Build & Run
```bash
# Build
docker build -t citadel-rl:latest docker/

# Run (done by tester.py automatically)
docker run --gpus all --cpus=15 --network none \
  -v output_dir:/workspace/output \
  -v code_file:/workspace/train.py:ro \
  citadel-rl:latest python /workspace/train.py
```

### Diagnostics
The Dockerfile includes a built-in diagnostics script at `/workspace/diagnostics.py` that checks:
- Python version, GPU availability, CUDA version
- All critical imports (gymnasium, stable_baselines3, torch, etc.)
- MuJoCo installation

### Notes
- `--cpus=15` in docker run (AMD 9800X3D: 8C/16T, leave 1 thread for host)
- Image name in tester.py: `DOCKER_IMAGE = "citadel-rl:latest"`
- Video recording requires `render_mode="rgb_array"` in gym.make()
