# RL Sandbox - CPU-only Python sandbox for reinforcement learning code execution
# Designed for safe execution of LLM-generated code

FROM python:3.11-slim

# Install system dependencies
# - ffmpeg: Required for gymnasium video recording
# - swig: Required for Box2D environments
# - build-essential: Required for compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    swig \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (smaller than GPU version)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install Python packages (CPU-only)
RUN pip install --no-cache-dir \
    gymnasium[classic-control,box2d]==0.29.1 \
    stable-baselines3 \
    numpy \
    matplotlib \
    pillow \
    moviepy

# Create non-root user for security
RUN useradd -m -u 1000 sandbox

# Set up workspace directory
RUN mkdir -p /workspace/output && chown -R sandbox:sandbox /workspace

# Switch to non-root user
USER sandbox

WORKDIR /workspace

# Default command (avoid 'code.py' name - causes circular import with pdb)
CMD ["python", "/workspace/agent_script.py"]
