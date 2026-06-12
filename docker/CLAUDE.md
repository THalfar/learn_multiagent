# docker/ - RL Training Sandbox

## Dockerfile — GPU-accelerated Python sandbox
Base image: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`

### Installed stack
- Python 3 + pip
- PyTorch 2.7 (CUDA 12.8)
- `stable-baselines3[extra]` (includes tensorboard, tqdm, rich)
- gymnasium + gymnasium[classic-control,box2d,mujoco]
- panda-gym (PyBullet robotic manipulation — goal-conditioned + sparse-reward HER tasks; `import panda_gym` registers PandaReach/Push/Slide/PickAndPlace/Stack-v3)
- gym-pybullet-drones (PyBullet quadrotor RL — hover/flight aviaries). Pinned to commit
  `cfa7af6` (Feb 2025), the LAST gymnasium-0.29-compatible revision: the Oct 2025 main
  bumped to gymnasium ^1.2 + numpy ^2.2 which conflicts with this stack. Installed from
  the GitHub tarball with `--no-deps` (so its numpy ^1.24 caret can't downgrade anything);
  only missing runtime dep `transforms3d` installed separately.
- sb3-contrib (TQC, QR-DQN, etc. — algorithm breadth for the general-intelligence runs)
- scipy, pandas, seaborn, optuna, opencv-python-headless
- MuJoCo (system-level install)

### Auto-import (.pth mechanism)
Both robotics suites are imported at EVERY interpreter start via `.pth` files
(`zz_panda_autoload.pth`, `zz_drones_autoload.pth`), so their envs are always registered —
generated code never needs to remember `import panda_gym` / `import gym_pybullet_drones`.

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
