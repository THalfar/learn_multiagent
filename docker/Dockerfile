# RL Sandbox - GPU-accelerated Python sandbox for reinforcement learning code execution
# RTX 5090 (Blackwell SM 12.0) - PyTorch 2.7 stable with CUDA 12.8

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install system dependencies (including MuJoCo requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    swig \
    build-essential \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch 2.7 stable with CUDA 12.8 (official Blackwell SM 12.0 support)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Install Python packages (including MuJoCo for Humanoid-v4 etc.)
# Core RL stack - "moderni" full-featured environment
RUN pip install --no-cache-dir \
    mujoco \
    gymnasium[classic-control,box2d,mujoco]==0.29.1 \
    'stable-baselines3[extra]' \
    numpy \
    matplotlib \
    pillow \
    moviepy \
    scipy \
    pandas \
    seaborn \
    optuna \
    opencv-python-headless

# Robotic manipulation: panda-gym (PyBullet) for goal-conditioned + sparse-reward
# tasks (the HER intelligence test). Separate layer so the big install above stays
# cached on rebuild. PyBullet renders headless (rgb_array) without MuJoCo egl setup.
RUN pip install --no-cache-dir panda-gym==3.0.7

# Auto-register panda-gym envs on EVERY Python startup, so generated training code never
# has to remember `import panda_gym`. A .pth file's "import" line runs at interpreter
# start. This makes gym.make("PandaReach-v3") work even if the script forgets the import.
RUN echo "import panda_gym" > /usr/local/lib/python3.10/dist-packages/zz_panda_autoload.pth

# Create non-root user for security
RUN useradd -m -u 1000 sandbox

# Set up workspace directory FIRST (before writing any files to it)
RUN mkdir -p /workspace/output && chown -R sandbox:sandbox /workspace

# Save installed packages list
RUN pip list > /workspace/installed_packages.txt

# Create diagnostic script for debugging imports
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
print("=== Python Environment Diagnostics ===")\n\
print(f"Python: {sys.version}")\n\
print()\n\
try:\n\
    import stable_baselines3 as sb3\n\
    print(f"stable-baselines3: {sb3.__version__}")\n\
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback\n\
    print("  EvalCallback: OK")\n\
    print("  CheckpointCallback: OK")\n\
    from stable_baselines3.common.evaluation import evaluate_policy\n\
    print("  evaluate_policy: OK")\n\
except Exception as e:\n\
    print(f"  ERROR: {e}")\n\
print()\n\
try:\n\
    import gymnasium\n\
    print(f"gymnasium: {gymnasium.__version__}")\n\
    from gymnasium.wrappers import RecordVideo\n\
    print("  RecordVideo: OK")\n\
except Exception as e:\n\
    print(f"  ERROR: {e}")\n\
print()\n\
try:\n\
    import torch\n\
    print(f"PyTorch: {torch.__version__}")\n\
    print(f"  CUDA available: {torch.cuda.is_available()}")\n\
    if torch.cuda.is_available():\n\
        print(f"  GPU: {torch.cuda.get_device_name(0)}")\n\
except Exception as e:\n\
    print(f"  ERROR: {e}")\n\
print()\n\
try:\n\
    import mujoco\n\
    print(f"MuJoCo: {mujoco.__version__}")\n\
except Exception as e:\n\
    print(f"MuJoCo: NOT INSTALLED ({e})")\n\
print()\n\
print("=== MuJoCo Environment Test ===")\n\
try:\n\
    import gymnasium\n\
    env = gymnasium.make("Humanoid-v4")\n\
    print("  Humanoid-v4: OK")\n\
    env.close()\n\
except Exception as e:\n\
    print(f"  Humanoid-v4: FAILED ({e})")\n\
print()\n\
print("=== Available SB3 Callbacks ===")\n\
try:\n\
    from stable_baselines3.common import callbacks\n\
    available = [x for x in dir(callbacks) if not x.startswith("_")]\n\
    for cb in available:\n\
        print(f"  {cb}")\n\
except Exception as e:\n\
    print(f"  ERROR: {e}")\n\
' > /workspace/diagnostics.py && chmod +x /workspace/diagnostics.py

# Switch to non-root user
USER sandbox

WORKDIR /workspace

# Default command
CMD ["python", "/workspace/agent_script.py"]
