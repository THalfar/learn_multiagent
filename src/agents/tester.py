import json
import os
import re
import subprocess
import platform
from pathlib import Path
from .base import BaseAgent
from rich import print
from rich.markup import escape as rich_escape

ALLOWED_DIR = os.path.abspath("output/")
PROJECT_ROOT = os.path.abspath(".")

# Docker sandbox configuration
DOCKER_SANDBOX_ENABLED = True  # Set to False to run locally (for development)
DOCKER_IMAGE = "rl-sandbox:latest"

# GPU validation flag - set to True after first successful validation
_GPU_VALIDATED = False


def extract_json(content):
    """Extract and parse JSON from LLM response, handling various formats."""
    content = content.strip()

    # Try to extract from markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()

    # Try to find JSON object in the content
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        content = content[json_start:json_end]

    return content


def get_gpu_vram_usage() -> dict:
    """
    Get current GPU VRAM usage using nvidia-smi.
    Returns dict with total, used, free in GB.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return {
                    "total_gb": float(parts[0]) / 1024,
                    "used_gb": float(parts[1]) / 1024,
                    "free_gb": float(parts[2]) / 1024,
                    "available": True
                }
    except Exception:
        pass
    return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "available": False}


def run_doc_inspection_in_container(class_path: str, timeout: int = 30) -> str:
    """
    Run documentation/inspection commands in the Docker container to check class signatures and help.

    Args:
        class_path: Full path to class like "gymnasium.wrappers.RecordVideo" or "stable_baselines3.PPO"
        timeout: Execution timeout in seconds

    Returns:
        Combined output with signature, docstring, and available methods
    """
    # Parse class path to get module and class name
    parts = class_path.rsplit('.', 1)
    if len(parts) == 2:
        module_path, class_name = parts
    else:
        return f"Invalid class path: {class_path}. Use format like 'gymnasium.wrappers.RecordVideo'"

    # Build inspection script - avoid f-string nesting issues by using .format() for outer variables
    inspection_script = '''
import inspect
import sys

CLASS_PATH = "''' + class_path + '''"
MODULE_PATH = "''' + module_path + '''"
CLASS_NAME = "''' + class_name + '''"

print("=" * 60)
print(f"DOCUMENTATION INSPECTION: {CLASS_PATH}")
print("=" * 60)
print()

try:
    # Dynamic import
    module_parts = MODULE_PATH.split(".")
    module = __import__(module_parts[0])
    for part in module_parts[1:]:
        module = getattr(module, part)

    cls = getattr(module, CLASS_NAME)

    print("=== CLASS SIGNATURE ===")
    try:
        sig = inspect.signature(cls)
        print(f"{CLASS_NAME}{sig}")
        print()
        print("Parameters:")
        for name, param in sig.parameters.items():
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation.__name__}" if param.annotation != inspect.Parameter.empty and hasattr(param.annotation, "__name__") else ""
            print(f"  - {name}{annotation}{default}")
    except Exception as e:
        print(f"Could not get signature: {e}")

    print()
    print("=== DOCSTRING (first 1500 chars) ===")
    doc = cls.__doc__ or "No docstring available"
    print(doc[:1500])
    if len(doc) > 1500:
        print("... (truncated)")

    print()
    print("=== PUBLIC METHODS ===")
    methods = [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m, None))]
    print(", ".join(methods[:30]))
    if len(methods) > 30:
        print(f"... and {len(methods) - 30} more")

    print()
    print("=== INIT METHOD SIGNATURE ===")
    try:
        init_sig = inspect.signature(cls.__init__)
        print(f"__init__{init_sig}")
    except Exception as e:
        print(f"Could not get __init__ signature: {e}")

except ImportError as e:
    print(f"Import error: {e}")
except AttributeError as e:
    print(f"Attribute error: {e}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()
print("=" * 60)
'''

    cmd = [
        "docker", "run",
        "--rm",
        "--network=none",
        "--memory=2g",
        "--cpus=2",
        DOCKER_IMAGE,
        "python", "-c", inspection_script
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        return result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return f"Documentation inspection timed out after {timeout}s"
    except Exception as e:
        return f"Documentation inspection error: {e}"


def run_diagnostic_in_container(timeout: int = 30) -> str:
    """
    Run diagnostic commands in the Docker container to check installed packages.

    Args:
        timeout: Execution timeout in seconds

    Returns:
        Combined stdout + stderr output with package versions
    """
    # Inline Python diagnostic script - no file needed
    diagnostic_script = '''
import sys
print("=== PYTHON ENVIRONMENT ===")
print(f"Python: {sys.version}")
print()
print("=== PACKAGE VERSIONS ===")
try:
    import gymnasium; print(f"gymnasium: {gymnasium.__version__}")
except ImportError as e: print(f"gymnasium: NOT INSTALLED ({e})")
try:
    import stable_baselines3; print(f"stable_baselines3: {stable_baselines3.__version__}")
except ImportError as e: print(f"stable_baselines3: NOT INSTALLED ({e})")
try:
    import torch; print(f"torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e: print(f"torch: NOT INSTALLED ({e})")
try:
    import numpy; print(f"numpy: {numpy.__version__}")
except ImportError as e: print(f"numpy: NOT INSTALLED ({e}}")
print()
print("=== SB3 SUBMODULES ===")
try:
    from stable_baselines3.common import vec_env
    print(f"stable_baselines3.common.vec_env: OK")
    print(f"  Available: {[x for x in dir(vec_env) if not x.startswith('_')]}")
except ImportError as e: print(f"stable_baselines3.common.vec_env: FAILED ({e})")
try:
    from stable_baselines3.common import callbacks
    print(f"stable_baselines3.common.callbacks: OK")
    print(f"  Available: {[x for x in dir(callbacks) if not x.startswith('_')]}")
except ImportError as e: print(f"stable_baselines3.common.callbacks: FAILED ({e})")
'''

    cmd = [
        "docker", "run",
        "--rm",
        "--network=none",
        "--memory=2g",
        "--cpus=2",
        DOCKER_IMAGE,
        "python", "-c", diagnostic_script
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace undecodable bytes instead of crashing
            timeout=timeout
        )
        return result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return f"Diagnostic command timed out after {timeout}s"
    except Exception as e:
        return f"Diagnostic error: {e}"


def validate_gpu_in_container(timeout: int = 60) -> tuple[bool, str]:
    """
    Validate GPU/CUDA functionality in Docker container before running any RL code.
    Runs a minimal PyTorch CUDA test + tiny RL training to ensure everything works.

    Args:
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    global _GPU_VALIDATED

    if _GPU_VALIDATED:
        return True, "GPU already validated"

    print("\n" + "─" * 70)
    print("[bold cyan]🔍 GPU/CUDA VALIDATION CHECK[/bold cyan]")
    print("─" * 70)
    print("[cyan]Running GPU validation before first RL execution...[/cyan]")

    # Comprehensive GPU + minimal RL validation script
    validation_script = '''
import sys
print("=" * 60)
print("GPU/CUDA VALIDATION TEST")
print("=" * 60)

# Step 1: Check PyTorch CUDA
print("\\n[1/4] Checking PyTorch CUDA...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("  ERROR: CUDA not available in PyTorch!")
        sys.exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Step 2: Simple CUDA tensor operation
print("\\n[2/4] Testing CUDA tensor operation...")
try:
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = torch.matmul(x, y)
    print(f"  Matrix multiplication on GPU: OK")
    print(f"  Result shape: {z.shape}")
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    del x, y, z
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  ERROR: CUDA tensor operation failed: {e}")
    sys.exit(1)

# Step 3: Check Stable Baselines3
print("\\n[3/4] Checking Stable Baselines3...")
try:
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    print(f"  SB3 version: {stable_baselines3.__version__}")
    print(f"  PPO import: OK")
    print(f"  DummyVecEnv import: OK")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Step 4: Minimal RL training on GPU
print("\\n[4/4] Running minimal RL training on GPU...")
try:
    import gymnasium as gym

    # Create simple environment
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # Create PPO model on CUDA
    model = PPO("MlpPolicy", env, device="cuda", verbose=0)
    print(f"  PPO model created on CUDA: OK")

    # Train for just 100 steps
    model.learn(total_timesteps=100, progress_bar=False)
    print(f"  Training 100 steps on GPU: OK")

    # Check GPU was actually used
    gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
    print(f"  GPU memory used during training: {gpu_mem:.1f} MB")

    env.close()
    del model
    torch.cuda.empty_cache()

    print("\\n" + "=" * 60)
    print("GPU VALIDATION: SUCCESS")
    print("=" * 60)

except Exception as e:
    print(f"  ERROR: Minimal RL training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    cmd = [
        "docker", "run",
        "--rm",
        "--gpus", "all",
        "--network=none",
        "--memory=4g",
        "--cpus=4",
        DOCKER_IMAGE,
        "python", "-c", validation_script
    ]

    try:
        print("[dim]   Running GPU validation in Docker container...[/dim]")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )

        output = result.stdout + "\n" + result.stderr

        if result.returncode == 0:
            _GPU_VALIDATED = True
            print("[bold green]✅ GPU VALIDATION PASSED[/bold green]")
            print("[green]   CUDA working, PyTorch OK, minimal RL training successful[/green]")
            print("─" * 70 + "\n")
            return True, output
        else:
            print("[bold red]❌ GPU VALIDATION FAILED[/bold red]")
            print(f"[red]   Return code: {result.returncode}[/red]")
            # Show error details
            error_lines = output.strip().split('\n')
            for line in error_lines[-10:]:  # Last 10 lines
                print(f"[dim red]   {line}[/dim red]")
            print("─" * 70 + "\n")
            return False, output

    except subprocess.TimeoutExpired:
        msg = f"GPU validation timed out after {timeout}s"
        print(f"[bold red]❌ {msg}[/bold red]")
        print("─" * 70 + "\n")
        return False, msg
    except Exception as e:
        msg = f"GPU validation error: {e}"
        print(f"[bold red]❌ {msg}[/bold red]")
        print("─" * 70 + "\n")
        return False, msg


def run_in_container(code_path: str, output_dir: str, timeout: int, gpu_enabled: bool = True) -> tuple[subprocess.CompletedProcess, dict]:
    """
    Execute code in Docker container for isolation with GPU support.

    Args:
        code_path: Absolute path to Python code file on host
        output_dir: Absolute path to output directory on host (for videos, logs)
        timeout: Execution timeout in seconds
        gpu_enabled: Whether to enable GPU access in container

    Returns:
        Tuple of (subprocess.CompletedProcess, vram_stats dict)
    """
    global _GPU_VALIDATED

    # Run GPU validation on first container execution (if GPU enabled)
    if gpu_enabled and not _GPU_VALIDATED:
        success, validation_msg = validate_gpu_in_container()
        if not success:
            print("[bold yellow]⚠️  GPU validation failed - continuing anyway but GPU may not work[/bold yellow]")
            print(f"[dim yellow]   {validation_msg[:200]}[/dim yellow]")

    # Convert Windows paths to absolute paths
    code_path_abs = os.path.abspath(code_path)
    output_dir_abs = os.path.abspath(output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    # Get VRAM usage BEFORE execution
    vram_before = get_gpu_vram_usage()

    cmd = [
        "docker", "run",
        "--rm",                              # Remove container after execution
        "--network=none",                    # No network access
        "--memory=8g",                       # Memory limit (enough for parallel envs)
        "--cpus=15",                         # CPU limit: 15 threads for fast RL simulation
        "--pids-limit=200",                  # Process limit
    ]

    # Add GPU access if enabled
    if gpu_enabled:
        cmd.extend(["--gpus", "all"])

    cmd.extend([
        "-v", f"{code_path_abs}:/workspace/agent_script.py:ro",  # Code file (avoid 'code.py' - circular import)
        "-v", f"{output_dir_abs}:/workspace/output:rw",          # Output directory (read-write)
        DOCKER_IMAGE,
        "python", "/workspace/agent_script.py"
    ])

    # Fun Docker status messages
    print(f"[dim cyan]🐳 Docker sandbox initializing...[/dim cyan]")
    print(f"[dim]   Image: {DOCKER_IMAGE}[/dim]")
    print(f"[dim]   Memory limit: 8GB | CPUs: 15 threads[/dim]")
    if gpu_enabled:
        print(f"[bold green]   🎮 GPU: ENABLED (--gpus all)[/bold green]")
        if vram_before["available"]:
            print(f"[dim]   VRAM before: {vram_before['used_gb']:.2f}/{vram_before['total_gb']:.2f} GB used[/dim]")
    else:
        print(f"[dim yellow]   GPU: disabled (CPU-only mode)[/dim yellow]")
    print(f"[dim]   Network: disabled (sandboxed)[/dim]")
    print(f"[dim]   Timeout: {timeout}s ({timeout//60}m {timeout%60}s)[/dim]")
    print(f"[dim cyan]🚀 Launching container...[/dim cyan]")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',  # Replace undecodable bytes instead of crashing (Windows cp1252 fix)
        timeout=timeout
    )

    # Get VRAM usage AFTER execution
    vram_after = get_gpu_vram_usage()

    # Calculate VRAM delta (how much RL used)
    vram_stats = {
        "before": vram_before,
        "after": vram_after,
        "rl_used_gb": max(0, vram_after["used_gb"] - vram_before["used_gb"]) if vram_before["available"] else 0,
        "gpu_enabled": gpu_enabled
    }

    return result, vram_stats

def auto_fix_common_issues(code: str) -> str:
    """
    Automatically fix common coding mistakes that LLMs make.
    Returns the fixed code.
    """
    lines = code.splitlines()
    imports_needed = set()
    has_os_import = False
    has_numpy_import = False
    has_time_import = False

    # Check what's already imported
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import os") or "from os" in stripped:
            has_os_import = True
        if stripped.startswith("import numpy") or "from numpy" in stripped:
            has_numpy_import = True
        if stripped.startswith("import time") or "from time" in stripped:
            has_time_import = True

    # Check what's used but not imported
    code_lower = code
    if not has_os_import and ("os.path" in code or "os.makedirs" in code or "os.environ" in code or "os.listdir" in code):
        imports_needed.add("import os")
    if not has_numpy_import and "np." in code:
        imports_needed.add("import numpy as np")
    if not has_time_import and "time.sleep" in code or "time.time" in code:
        imports_needed.add("import time")

    # If fixes needed, insert them at the top
    if imports_needed:
        print(f"[yellow]🔧 Auto-fixing missing imports: {', '.join(imports_needed)}[/yellow]")
        # Find the first import line to insert after
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                insert_idx = i + 1
                break

        # Insert missing imports
        for imp in sorted(imports_needed):
            lines.insert(insert_idx, imp)
            insert_idx += 1

        return "\n".join(lines)

    return code


def is_safe_code(code: str) -> bool:
    """
    Check if code is safe to execute.
    Blocks dangerous operations while allowing safe video directory operations.
    """
    # Critical system paths to block
    dangerous_patterns = [
        "rm -rf",           # Shell command for recursive delete
        "rmdir /s",         # Windows recursive delete
        "C:\\\\Windows\\\\",  # Windows system directory
        "C:\\\\Program Files",  # Program files directory
        "/etc/",            # Linux system config
        "/sys/",            # Linux system files
        "/proc/",           # Linux process info
        "subprocess",       # Subprocess execution
        "import os\\nos.system",  # Direct os.system calls
        "exec(",            # Dynamic code execution
        "eval(",            # Dynamic code evaluation
        "__import__",       # Dynamic imports
    ]
    
    # Block dangerous patterns
    for pattern in dangerous_patterns:
        if pattern in code:
            return False
    
    # Allow shutil.rmtree ONLY with ignore_errors=True or within try-except
    if "shutil.rmtree" in code:
        # Check if it has ignore_errors=True
        if "ignore_errors=True" not in code:
            # Check if it's in a try-except block (simple heuristic)
            if "try:" not in code or "except" not in code:
                return False
    
    # Allow os.remove only for specific file patterns (like .zip, .pkl, .csv, etc.)
    if "os.remove" in code:
        # Check if it's removing a specific file (not a directory)
        # This is a simple check - you might want to make it more sophisticated
        safe_file_patterns = [".zip", ".pkl", ".csv", ".log", ".txt", ".json"]
        has_safe_pattern = any(pattern in code for pattern in safe_file_patterns)
        if not has_safe_pattern:
            return False
    
    # Block attempts to navigate outside project with relative paths
    if "../../../" in code or "..\\..\\.\\" in code:  # More than 2 levels up
        return False
    
    return True

def check_video_files(video_dir: str) -> dict:
    """
    Check if video files exist in the video directory and validate them.
    Returns a dict with video file information.
    """
    video_info = {
        "video_dir": video_dir,
        "exists": False,
        "video_files": [],
        "total_size": 0,
        "valid_videos": 0,
        "empty_videos": 0,
        "error": None
    }
    
    try:
        video_path = Path(video_dir)
        if not video_path.exists():
            video_info["error"] = f"Video directory does not exist: {video_dir}"
            return video_info
        
        video_info["exists"] = True
        
        # Find all video files (common formats) - RECURSIVE SEARCH
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []

        # Recursive search for all video files in the entire directory tree
        # This catches videos in any subdirectory structure like:
        # - videos/iter_X/envY/*.mp4
        # - iter_X/*.mp4
        # - rl-video-*.mp4
        for ext in video_extensions:
            video_files.extend(list(video_path.rglob(f'*{ext}')))

        # Also check for rl-video files recursively
        rl_video_files = list(video_path.rglob('rl-video-*'))
        rl_video_files = [f for f in rl_video_files if f.suffix in video_extensions]
        video_files.extend(rl_video_files)
        
        # Remove duplicates and filter to actual video files only
        video_files = list(set([f for f in video_files if f.suffix in video_extensions]))
        
        if not video_files:
            video_info["error"] = f"No video files found in {video_dir}"
            return video_info
        
        # Check each video file
        for video_file in sorted(video_files):
            file_info = {
                "path": str(video_file),
                "name": video_file.name,
                "size": 0,
                "size_mb": 0.0,
                "is_valid": False,
                "is_empty": False
            }
            
            try:
                if video_file.exists():
                    file_size = video_file.stat().st_size
                    file_info["size"] = file_size
                    file_info["size_mb"] = file_size / (1024 * 1024)
                    
                    # Check if file is empty (likely not a real video)
                    if file_size == 0:
                        file_info["is_empty"] = True
                        video_info["empty_videos"] += 1
                    elif file_size < 1024:  # Less than 1KB is suspicious
                        file_info["is_empty"] = True
                        video_info["empty_videos"] += 1
                    else:
                        # Check if it's a valid MP4 by reading file header
                        # MP4 files start with specific bytes (ftyp box)
                        try:
                            with open(video_file, 'rb') as f:
                                header = f.read(12)
                                # MP4 files have 'ftyp' at offset 4
                                if len(header) >= 8 and header[4:8] == b'ftyp':
                                    file_info["is_valid"] = True
                                    video_info["valid_videos"] += 1
                                elif video_file.suffix == '.mp4':
                                    # If it's .mp4 but doesn't have proper header, might be corrupted
                                    file_info["is_valid"] = False
                                else:
                                    # For other formats, assume valid if size is reasonable
                                    file_info["is_valid"] = True
                                    video_info["valid_videos"] += 1
                        except Exception as e:
                            # Can't read file, mark as invalid
                            file_info["error"] = str(e)
                    video_info["total_size"] += file_size
            except Exception as e:
                file_info["error"] = str(e)
            
            video_info["video_files"].append(file_info)
        
    except Exception as e:
        video_info["error"] = f"Error checking video directory: {str(e)}"
    
    return video_info

class Tester(BaseAgent):
    def __init__(self, config, model_switcher=None):
        super().__init__(config, "tester", model_switcher=model_switcher)

    def __call__(self, state: dict) -> dict:
        # NOTE: No transition banner here - coder doesn't have opinions so nothing to show
        code = state.get("code", "")

        # Remove invalid 'timeout' argument from model.learn() call in generated code
        code = re.sub(
            r'timeout\\s*=\\s*[^,\\)]+(?=\\s*[,\\)]|$)',
            '',
            code,
            flags=re.IGNORECASE
        )
        if not code:
            return {
                "test_results": "No code to test",
                "execution_stdout": "",
                "execution_stderr": ""
            }

        if not is_safe_code(code):
            return {
                "test_results": "ERROR: Dangerous code detected",
                "execution_stdout": "",
                "execution_stderr": "Dangerous code detected, execution blocked"
            }

        # Get execution timeout from current environment
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
        base_timeout = current_env.execution_timeout if current_env else (env_progression[0].execution_timeout if env_progression else 900)

        # MONIVAIHEINEN TREENI: Phase-based timeout
        current_phase = state.get("current_phase", "validation")
        training_phases = getattr(self.config.project, 'training_phases', None)

        if current_phase == "validation":
            # Validation: lyhyt timeout (5% normaalista tai konfiguraatiosta)
            multiplier = getattr(training_phases, 'validation_timeout_multiplier', 0.05) if training_phases else 0.05
            execution_timeout = max(60, int(base_timeout * multiplier))  # Min 60s
            print(f"[bold cyan]🔬 VALIDATION PHASE: Quick test (timeout: {execution_timeout}s = {multiplier*100:.0f}% of {base_timeout}s)[/bold cyan]")
        elif current_phase == "optimization":
            # Optimization: täysi timeout
            execution_timeout = base_timeout
            print(f"[bold green]🚀 OPTIMIZATION PHASE: Full training (timeout: {execution_timeout}s)[/bold green]")
        elif current_phase == "demo":
            # Demo: lyhyt timeout (5 min tai konfiguraatiosta)
            demo_timeout = getattr(training_phases, 'demo_timeout_seconds', 300) if training_phases else 300
            execution_timeout = demo_timeout
            print(f"[bold magenta]🎬 DEMO PHASE: Video recording (timeout: {execution_timeout}s)[/bold magenta]")
        else:
            execution_timeout = base_timeout
        
        run_id = state["run_id"]
        # Use env-specific subdirectory for code and videos
        current_env_name = current_env.name if current_env else "unknown"
        code_dir = f"output/{run_id}/{current_env_name}/code"
        video_dir = state["video_dir"]
        
        # Normalize video_dir to absolute path (fixes Windows path issues)
        import os
        video_dir = os.path.abspath(os.path.normpath(video_dir))
        
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        code_path = f"{code_dir}/agent_code_iter_{state.get('iteration', 0)}.py"

        print("\n\n" + "─" * 70)
        print("[bold yellow]🧪 TESTER: Waking up...[/bold yellow]")
        print("─" * 70)
        print("[yellow]💭 \"Another day, another test run. Let's see what the Coder cooked up this time...\"[/yellow]")
        print()
        print("[yellow]📝 Preparing sandbox environment...[/yellow]")
        print(f"[dim]   Code file: {code_path}[/dim]")
        print(f"[dim]   Video output: {video_dir}[/dim]")
        print(f"[dim]   Timeout: {execution_timeout}s ({execution_timeout//60}m {execution_timeout%60}s)[/dim]")
        print("[yellow]🔒 I can't see the code - I judge by results only[/yellow]")

        # Check if reviewer has a special request
        reviewer_instruction = state.get("reviewer_tester_instruction", None)
        doc_inspection_results = ""
        if reviewer_instruction:
            print()
            print("[bold cyan]📋 SHODAN's Command:[/bold cyan]")
            print(f"[cyan]   \"{reviewer_instruction}\"[/cyan]")
            print("[yellow]💭 \"The machine god has spoken. I shall investigate...\"[/yellow]")

            # Check if SHODAN wants documentation inspection
            # Look for patterns like "check help", "inspect", "documentation", "signature", class names
            doc_patterns = [
                r'check\s+(?:help|docs?|documentation|signature)',
                r'inspect\s+\w+',
                r'what\s+(?:are|is)\s+the\s+(?:parameters?|arguments?|signature)',
                r'help\s*\(\s*\w+',
                r'RecordVideo|EvalCallback|PPO|A2C|DQN|SAC|Monitor|DummyVecEnv',
            ]

            # Extract class names from instruction
            class_patterns = [
                (r'gymnasium\.wrappers\.(\w+)', 'gymnasium.wrappers.{}'),
                (r'gymnasium\.(\w+)', 'gymnasium.{}'),
                (r'stable_baselines3\.common\.callbacks\.(\w+)', 'stable_baselines3.common.callbacks.{}'),
                (r'stable_baselines3\.common\.vec_env\.(\w+)', 'stable_baselines3.common.vec_env.{}'),
                (r'stable_baselines3\.common\.evaluation\.(\w+)', 'stable_baselines3.common.evaluation.{}'),
                (r'stable_baselines3\.(\w+)', 'stable_baselines3.{}'),
                (r'\b(RecordVideo)\b', 'gymnasium.wrappers.{}'),
                (r'\b(EvalCallback|CheckpointCallback|BaseCallback)\b', 'stable_baselines3.common.callbacks.{}'),
                (r'\b(DummyVecEnv|SubprocVecEnv|VecMonitor)\b', 'stable_baselines3.common.vec_env.{}'),
                (r'\b(evaluate_policy)\b', 'stable_baselines3.common.evaluation.{}'),
                (r'\b(PPO|A2C|DQN|SAC)\b', 'stable_baselines3.{}'),
                (r'\b(Monitor)\b', 'gymnasium.wrappers.{}'),
            ]

            classes_to_inspect = []
            for pattern, template in class_patterns:
                matches = re.findall(pattern, reviewer_instruction, re.IGNORECASE)
                for match in matches:
                    class_path = template.format(match)
                    if class_path not in classes_to_inspect:
                        classes_to_inspect.append(class_path)

            # Also check for generic doc request keywords
            needs_doc_check = any(re.search(p, reviewer_instruction, re.IGNORECASE) for p in doc_patterns)

            if classes_to_inspect or needs_doc_check:
                print()
                print("[bold green]📚 DOCUMENTATION INSPECTION TRIGGERED[/bold green]")

                if not classes_to_inspect:
                    # Default classes to check if generic request
                    classes_to_inspect = ['gymnasium.wrappers.RecordVideo']

                for class_path in classes_to_inspect[:3]:  # Max 3 classes
                    print(f"[green]   Inspecting: {class_path}[/green]")
                    inspection_result = run_doc_inspection_in_container(class_path)
                    doc_inspection_results += f"\n\n{inspection_result}"
                    # Escape Rich markup in the output (e.g., [int] in Callable[[int], bool])
                    escaped_result = rich_escape(inspection_result)
                    print(f"[dim]{escaped_result[:500]}...[/dim]" if len(escaped_result) > 500 else f"[dim]{escaped_result}[/dim]")

                print("[green]📚 Documentation inspection complete[/green]")
        else:
            print("[yellow]No special instructions from SHODAN this time.[/yellow]")
        print("─" * 70 + "\n")

        # Auto-fix common LLM mistakes before execution
        code = auto_fix_common_issues(code)

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # Run with environment-specific timeout
            import time
            execution_start = time.time()

            if DOCKER_SANDBOX_ENABLED:
                # Get GPU settings from config
                gpu_enabled = self.config.gpu.enabled
                max_vram_gb = self.config.gpu.max_vram_gb
                warn_vram_gb = self.config.gpu.warn_vram_gb

                # Run in Docker container for isolation (with GPU if enabled)
                result, vram_stats = run_in_container(code_path, video_dir, execution_timeout, gpu_enabled)
            else:
                # Run locally (for development/debugging) - no VRAM tracking
                cmd = ["python", code_path]
                result = subprocess.run(
                    cmd,
                    cwd=os.getcwd(),
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # Replace undecodable bytes (Windows fix)
                    timeout=execution_timeout,
                )
                # Dummy VRAM stats for local execution
                vram_stats = {"gpu_enabled": False, "before": {"available": False}, "after": {"available": False}, "rl_used_gb": 0}
                max_vram_gb = 4.0
                warn_vram_gb = 2.0

            execution_end = time.time()
            execution_duration = execution_end - execution_start

            stdout = result.stdout
            stderr = result.stderr

            # If ImportError detected, run diagnostics automatically
            if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
                print("\n[bold yellow]🔍 ImportError detected - running container diagnostics...[/bold yellow]")
                diag_output = run_diagnostic_in_container()  # No file needed - runs inline Python
                print(f"[dim cyan]{diag_output}[/dim cyan]")
                # Append diagnostics to stderr for LLM to see
                stderr = stderr + "\n\n=== CONTAINER DIAGNOSTICS ===\n" + diag_output

            # Print VRAM statistics (GPU usage by RL model)
            if vram_stats["gpu_enabled"] and vram_stats["before"]["available"]:
                print("\n" + "─" * 70)
                print("[bold cyan]🎮 GPU/VRAM STATISTICS[/bold cyan]")
                print("─" * 70)
                print(f"[dim]   VRAM before RL: {vram_stats['before']['used_gb']:.2f} GB[/dim]")
                print(f"[dim]   VRAM after RL:  {vram_stats['after']['used_gb']:.2f} GB[/dim]")
                rl_vram = vram_stats["rl_used_gb"]
                if rl_vram > 0:
                    if rl_vram > max_vram_gb:
                        print(f"[bold red]   ⚠️  RL model VRAM: {rl_vram:.2f} GB (EXCEEDS LIMIT: {max_vram_gb} GB!)[/bold red]")
                    elif rl_vram > warn_vram_gb:
                        print(f"[yellow]   ⚠️  RL model VRAM: {rl_vram:.2f} GB (warning: >{warn_vram_gb} GB)[/yellow]")
                    else:
                        print(f"[green]   ✓ RL model VRAM: {rl_vram:.2f} GB (limit: {max_vram_gb} GB)[/green]")
                else:
                    print(f"[dim]   RL model VRAM: ~0 GB (minimal/released)[/dim]")
                print(f"[dim]   Total GPU VRAM: {vram_stats['before']['total_gb']:.1f} GB[/dim]")
                print("─" * 70)

                # Add VRAM info to stderr for LLM analysis
                vram_info = f"\n=== GPU VRAM USAGE ===\nRL model used: {rl_vram:.2f} GB\nLimit: {max_vram_gb} GB\nStatus: {'OK' if rl_vram <= max_vram_gb else 'EXCEEDED'}\n"
                stderr = (stderr or "") + vram_info

            # Check for video files after execution
            video_check = check_video_files(video_dir)
            
            # Add video check information to output
            video_check_output = []
            video_check_output.append("\n" + "─" * 70)
            video_check_output.append("[bold cyan]🎬 VIDEO FILE CHECK[/bold cyan]")
            video_check_output.append("─" * 70)
            video_check_output.append(f"[dim]Scanning: {video_dir}[/dim]")
            
            if video_check["error"]:
                video_check_output.append(f"[red]❌ Error: {video_check['error']}[/red]")
            elif not video_check["exists"]:
                video_check_output.append(f"[red]📁 Video directory does not exist: {video_dir}[/red]")
                video_check_output.append(f"[dim]   (Coder needs to add RecordVideo wrapper)[/dim]")
            elif not video_check["video_files"]:
                video_check_output.append(f"[yellow]🔍 No video files found in {video_dir}[/yellow]")
                video_check_output.append(f"[dim]   (Training ran but no video recorded)[/dim]")
            else:
                video_check_output.append(f"[green]🎥 Found {len(video_check['video_files'])} video file(s)![/green]")
                video_check_output.append(f"[dim]   Total size: {video_check['total_size'] / (1024*1024):.2f} MB[/dim]")
                video_check_output.append(f"[green]   ✓ Valid videos: {video_check['valid_videos']}[/green]")
                if video_check["empty_videos"] > 0:
                    video_check_output.append(f"[red]   ✗ Empty/corrupted: {video_check['empty_videos']}[/red]")
                if video_check['valid_videos'] > 0:
                    video_check_output.append(f"[dim cyan]   🎬 Agent behavior captured on film![/dim cyan]")
                
                # Show details of first few videos
                for i, vf in enumerate(video_check["video_files"][:5]):
                    status = "✓" if vf["is_valid"] and not vf["is_empty"] else "✗"
                    color = "green" if vf["is_valid"] and not vf["is_empty"] else "red"
                    video_check_output.append(f"[{color}]{status} {vf['name']}: {vf['size_mb']:.2f} MB[/{color}]")
                
                if len(video_check["video_files"]) > 5:
                    video_check_output.append(f"[dim]... and {len(video_check['video_files']) - 5} more[/dim]")
            
            video_check_output.append("─" * 70 + "\n")
            
            # Print video check results
            for line in video_check_output:
                print(line)
            
            # Add video check info to stderr for LLM analysis
            video_info_text = "\n".join([
                "=== VIDEO FILE CHECK ===",
                f"Video directory: {video_dir}",
                f"Directory exists: {video_check['exists']}",
                f"Video files found: {len(video_check['video_files'])}",
                f"Valid videos: {video_check['valid_videos']}",
                f"Empty/corrupted: {video_check['empty_videos']}",
                f"Total size: {video_check['total_size'] / (1024*1024):.2f} MB" if video_check['total_size'] > 0 else "Total size: 0 MB",
            ])
            if video_check["error"]:
                video_info_text += f"\nError: {video_check['error']}"
            if video_check["video_files"]:
                video_info_text += "\nVideo files:"
                for vf in video_check["video_files"][:10]:  # Limit to first 10
                    status = "VALID" if vf["is_valid"] and not vf["is_empty"] else "INVALID/EMPTY"
                    video_info_text += f"\n  - {vf['name']}: {vf['size_mb']:.2f} MB ({status})"
            video_info_text += "\n"
            
            # Append video check info to stderr so LLM can see it
            stderr = stderr + "\n" + video_info_text if stderr else video_info_text
            
            # Continue with analysis even if execution failed
            execution_failed = result.returncode != 0

            # Trigger adaptive model switch on repeated errors
            if execution_failed and self.model_switcher:
                from src.utils.model_switcher import SwitchTrigger
                error_msg = (result.stderr[:300] or result.stdout[:300]).strip()
                new_model = self.model_switcher.check_and_switch(
                    self.agent_name,
                    SwitchTrigger.REPEATED_ERROR,
                    {"error": error_msg}
                )
                if new_model:
                    self.switch_model(new_model)
            elif not execution_failed and self.model_switcher:
                # Success - reset error counters
                self.model_switcher.report_success(self.agent_name)

            # Show execution statistics
            if execution_failed:
                print("\n" + "─" * 70)
                print("[bold red]❌ EXECUTION FAILED[/bold red]")
                print("─" * 70)
                print(f"[red]Return code: {result.returncode}[/red]")
                print(f"[dim]Container terminated with error[/dim]")
                # Show brief error preview
                error_preview = (result.stderr[:300] or result.stdout[:300]).strip()
                if error_preview:
                    print(f"[dim]Error preview: {error_preview}...[/dim]")
            else:
                print("\n" + "─" * 70)
                print("[bold green]✅ EXECUTION COMPLETE[/bold green]")
                print("─" * 70)
                print(f"[dim]Container exited successfully (code 0)[/dim]")
            
            minutes = int(execution_duration // 60)
            seconds = int(execution_duration % 60)
            if minutes > 0:
                print(f"[green]⏱️  Execution time: {minutes}m {seconds}s ({execution_duration:.1f}s total)[/green]")
            else:
                print(f"[green]⏱️  Execution time: {seconds}s ({execution_duration:.1f}s total)[/green]")

            # Fun performance commentary
            if execution_duration < 30:
                print(f"[dim cyan]   ⚡ Lightning fast! The coder's algorithm is efficient.[/dim cyan]")
            elif execution_duration < 120:
                print(f"[dim cyan]   🏃 Good pace! Training is progressing well.[/dim cyan]")
            elif execution_duration < 300:
                print(f"[dim cyan]   🐢 Taking its time... complex environment perhaps?[/dim cyan]")
            else:
                print(f"[dim cyan]   🦥 Long training session! Hope the rewards are worth it.[/dim cyan]")
            
            # Try to extract training timesteps from output if available
            timesteps_match = re.search(r'total_timesteps[:\s]*(\d+)', stdout, re.IGNORECASE)
            if timesteps_match:
                timesteps = int(timesteps_match.group(1))
                print(f"[dim]Training timesteps: {timesteps:,}[/dim]")
            
            # Try to extract algorithm info
            algo_match = re.search(r'(PPO|A2C|DQN|SAC)', stdout, re.IGNORECASE)
            if algo_match:
                print(f"[dim]Algorithm: {algo_match.group(1)}[/dim]")
            
            print("─" * 70 + "\n")

            # LLM tester analysis - let LLM analyze all outputs (including errors)
            print("\n\n" + "─" * 70)
            print("[bold yellow]🧪 TESTER: Analyzing results...[/bold yellow]")
            print("─" * 70)
            print("[yellow]💭 \"Time to make sense of all this output...\"[/yellow]")
            print()
            print("[yellow]🤖 Engaging analysis mode...[/yellow]")
            print(f"[dim]   Model: {self.model_name}[/dim]")
            print(f"[dim]   Processing stdout ({len(stdout):,} chars) + stderr ({len(result.stderr):,} chars)[/dim]")

            # Show brief preview of what we're analyzing
            if stdout:
                stdout_preview = stdout[:200].replace('\n', ' ').strip()
                print(f"[dim]   stdout preview: \"{stdout_preview}...\"[/dim]")
            if result.stderr:
                stderr_preview = result.stderr[:150].replace('\n', ' ').strip()
                print(f"[dim]   stderr preview: \"{stderr_preview}...\"[/dim]")
            print()
            
            # Get success threshold from current environment (execution_timeout already retrieved earlier)
            env_progression = self.config.environment_progression
            current_env_index = state.get("current_env_index", 0)
            current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
            success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)
            
            prompt_dict = self.config.get_prompt("tester")

            # Get reviewer's special instruction from previous iteration (if any)
            reviewer_instruction = state.get("reviewer_tester_instruction", None)
            if reviewer_instruction:
                print(f"\n[cyan]📋 Reviewer's special request: {reviewer_instruction}[/cyan]\n")

            # Get agent opinions context (team chatter)
            agent_opinions_context = self.format_agent_opinions_context(state)

            # === DUAL LLM CALL OPTIMIZATION ===
            # If SHODAN requested documentation inspection AND we have results,
            # run TWO separate LLM calls for efficiency (model already in memory):
            # 1. Documentation analysis (focused, fast)
            # 2. Normal test analysis (execution results)
            doc_analysis_result = None
            if doc_inspection_results and reviewer_instruction:
                print("\n" + "─" * 70)
                print("[bold cyan]📚 DOCUMENTATION ANALYSIS (separate LLM call)[/bold cyan]")
                print("─" * 70)
                print("[cyan]Running focused documentation analysis first...[/cyan]")

                # Get doc analysis template
                doc_analysis_template = prompt_dict.get("doc_analysis_template", None)
                if doc_analysis_template:
                    doc_prompt_vars = {
                        "reviewer_instruction": reviewer_instruction,
                        "doc_inspection_results": doc_inspection_results,
                    }
                    doc_prompt = prompt_dict["system"] + "\n\n" + doc_analysis_template.format(**doc_prompt_vars)

                    # First LLM call - documentation analysis
                    doc_response = self.call_llm_timed(doc_prompt, state["stats"], state.get("iteration", 0))

                    # Parse doc analysis response
                    try:
                        doc_json = extract_json(doc_response.content)
                        doc_analysis_result = json.loads(doc_json)
                        doc_answer = doc_analysis_result.get("doc_answer", "")

                        print(f"[green]📖 Documentation answer:[/green]")
                        print(f"[dim]{doc_answer[:500]}{'...' if len(doc_answer) > 500 else ''}[/dim]")

                        if doc_analysis_result.get("correct_params"):
                            print(f"[green]✓ Correct parameters: {', '.join(doc_analysis_result['correct_params'])}[/green]")
                        if doc_analysis_result.get("common_mistakes"):
                            print(f"[yellow]⚠ Common mistakes: {', '.join(doc_analysis_result['common_mistakes'])}[/yellow]")
                    except json.JSONDecodeError:
                        doc_analysis_result = {"doc_answer": doc_response.content, "raw": True}
                        print(f"[dim]{doc_response.content[:500]}[/dim]")

                    print("─" * 70)
                    print("[cyan]Now running normal test analysis...[/cyan]\n")

            # Format template for normal test analysis
            # If we did separate doc analysis, don't include doc results in main prompt (already handled)
            template_vars = {
                "execution_stdout": stdout,
                "execution_stderr": stderr,
                "success_threshold": success_threshold,
                "manager_task": state.get("current_task", "No task description available"),
                "reviewer_instruction": reviewer_instruction or "None - no special request from reviewer",
                "agent_opinions_context": agent_opinions_context,
                "doc_inspection_results": "Documentation analysis completed separately - see results above." if doc_analysis_result else (doc_inspection_results if doc_inspection_results else "No documentation inspection performed."),
            }
            task_template = prompt_dict["task_template"].format(**template_vars)

            # Add conversation history (siloed - only this agent's previous messages)
            history_text = self.format_conversation_history(state)

            full_prompt = prompt_dict["system"] + "\n\n" + history_text + task_template

            # Print context breakdown before LLM call
            prompt_tokens = self.estimate_tokens(full_prompt)
            self.print_context_breakdown(state, prompt_tokens, agent_opinions_context)

            # Second LLM call (or only call if no doc analysis) - normal test analysis
            response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))
            
            # Print thinking process if using reasoning model
            self.print_thinking(response.content)
            
            # Print token statistics
            stats_obj = state["stats"]
            iteration = state.get("iteration", 0)
            agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == iteration]
            if agent_timings:
                latest_timing = agent_timings[-1]
                self.print_token_stats(latest_timing)

            try:
                json_content = extract_json(response.content)
                parsed = json.loads(json_content)
                summary = parsed.get("summary", "LLM analysis completed")
                tester_opinion = parsed.get("tester_opinion", "")
                success = parsed.get("success", None)
                metrics = parsed.get("metrics", {})
                my_opinion = parsed.get("my_opinion", "")  # Tester's personal take
                reviewer_response = parsed.get("reviewer_response", "")  # Response to SHODAN's request
                thoughts = parsed.get("thoughts", "")  # Tester's internal thoughts

                # Analysis output - this is the tester's report to reviewer
                print("\n" + "─" * 70)
                print("[bold yellow]🧪 TESTER'S ANALYSIS COMPLETE[/bold yellow]")
                print("─" * 70)

                # Show tester's thoughts/reasoning (if provided)
                if thoughts:
                    print("[yellow]💭 Tester's thoughts:[/yellow]")
                    print(f"[dim italic]   \"{thoughts}\"[/dim italic]")
                    print()

                # Main summary
                print(f"[bold]Summary:[/bold] {summary}")
                if success is not None:
                    if success:
                        print(f"[bold green]✅ Result: SUCCESS[/bold green]")
                    else:
                        print(f"[bold red]❌ Result: FAILED[/bold red]")

                # Metrics
                if metrics:
                    print()
                    print("[bold]📊 Metrics:[/bold]")
                    mean = metrics.get("mean_reward")
                    threshold = metrics.get("meets_threshold", False)
                    if mean is not None:
                        # Get success threshold from current environment
                        env_progression = self.config.environment_progression
                        current_env_index = state.get("current_env_index", 0)
                        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
                        success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)

                        if threshold:
                            print(f"[bold green]   Mean Reward: {mean} (threshold: {success_threshold}) ✓[/bold green]")
                        else:
                            print(f"[yellow]   Mean Reward: {mean} (threshold: {success_threshold}) ✗[/yellow]")
                    if metrics.get("std_reward") is not None:
                        print(f"[dim]   Std Reward: {metrics.get('std_reward')}[/dim]")
                    if metrics.get("n_episodes") is not None:
                        print(f"[dim]   Episodes: {metrics.get('n_episodes')}[/dim]")
                    if metrics.get("video_path"):
                        print(f"[dim]   Video: {metrics.get('video_path')}[/dim]")

                # Professional assessment for reviewer
                if tester_opinion:
                    print()
                    print(f"[yellow]📋 Assessment for SHODAN:[/yellow]")
                    print(f"   [yellow]{tester_opinion}[/yellow]")

                # Response to reviewer's special request (if any)
                # Combine doc analysis result with reviewer_response if both exist
                if doc_analysis_result and doc_analysis_result.get("doc_answer"):
                    print()
                    print("[bold cyan]📚 Documentation Analysis Result (from separate LLM call):[/bold cyan]")
                    print(f"[cyan]   {doc_analysis_result['doc_answer']}[/cyan]")
                    # Combine with any additional reviewer_response
                    if reviewer_response:
                        reviewer_response = f"DOCUMENTATION: {doc_analysis_result['doc_answer']}\n\nADDITIONAL: {reviewer_response}"
                    else:
                        reviewer_response = f"DOCUMENTATION: {doc_analysis_result['doc_answer']}"

                if reviewer_response:
                    print()
                    print("[bold cyan]📬 Response to SHODAN's Request:[/bold cyan]")
                    print(f"[cyan]   {reviewer_response[:500]}{'...' if len(reviewer_response) > 500 else ''}[/cyan]")
                elif reviewer_instruction:
                    print()
                    print("[dim cyan]📬 SHODAN asked: \"{reviewer_instruction}\"[/dim cyan]")
                    print("[dim]   (No specific response provided)[/dim]")

                # Team chatter (my_opinion)
                if my_opinion:
                    print()
                    print("[yellow]💬 Tester's take (team chatter):[/yellow]")
                    print(f"[yellow italic]   \"{my_opinion}\"[/yellow italic]")

                print("─" * 70 + "\n")
                
                # Combine summary, opinion and reviewer response for the report
                test_results_parts = [summary]
                if tester_opinion:
                    test_results_parts.append(f"Tester's assessment: {tester_opinion}")
                if reviewer_response:
                    test_results_parts.append(f"Response to SHODAN's request: {reviewer_response}")
                test_results = "\n\n".join(test_results_parts)

                # Remove any thinking tags from test_results (shouldn't be there, but safety check)
                test_results = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', test_results, flags=re.DOTALL | re.IGNORECASE)
                test_results = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', test_results, flags=re.DOTALL | re.IGNORECASE)
                test_results = test_results.strip()
            except json.JSONDecodeError as e:
                print(f"[bold red]ERROR: Tester LLM JSON parse failed: {e}[/bold red]")
                test_results = f"LLM analysis failed: {str(e)[:200]}"

            # Log to conversation file
            logger = state.get("conversation_logger")
            if logger:
                logger.log_tester(
                    iteration=state.get("iteration", 0),
                    test_results=test_results,
                    execution_stdout=stdout,
                    execution_stderr=result.stderr,
                    execution_time=execution_duration
                )

            # Save tester's response to conversation history
            history_update = self.save_message_to_history(state, response.content)

            # Save tester's opinion to state for team chatter
            opinion_update = self.save_opinion_to_state(state, my_opinion) if my_opinion else {}

            result_dict = {
                "test_results": test_results,
                "execution_stdout": stdout,
                "execution_stderr": result.stderr,
                "tester_reviewer_response": reviewer_response  # Tester's response to SHODAN's request
            }
            result_dict.update(history_update)
            result_dict.update(opinion_update)

            return result_dict

        except subprocess.TimeoutExpired:
            timeout_minutes = execution_timeout // 60
            timeout_seconds = execution_timeout % 60
            timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"

            # Trigger adaptive model switch on timeout
            if self.model_switcher:
                from src.utils.model_switcher import SwitchTrigger
                new_model = self.model_switcher.check_and_switch(
                    self.agent_name,
                    SwitchTrigger.TIMEOUT,
                    {"timeout_seconds": execution_timeout}
                )
                if new_model:
                    self.switch_model(new_model)

            print("\n\n" + "─" * 70)
            print(f"[bold yellow]🧪 TESTER: Timeout Report[/bold yellow]")
            print("─" * 70)
            print()
            print(f"[bold red]⏰ TIMEOUT: Execution exceeded {timeout_str} ({execution_timeout}s)[/bold red]")
            print()
            print(f"[yellow]🐳 Docker container forcibly terminated[/yellow]")
            print(f"[yellow]   The training took too long - try fewer timesteps or simpler approach[/yellow]")
            print()
            print(f"[yellow]📋 Assessment for SHODAN:[/yellow]")
            print(f"   [yellow]The code execution exceeded the timeout limit of {timeout_str}.[/yellow]")
            print(f"   [yellow]No reward metrics or video files could be generated.[/yellow]")
            print(f"   [yellow]Recommendation: Reduce training timesteps or simplify the training loop.[/yellow]")
            print()
            print("─" * 70 + "\n")

            # Log to conversation file
            logger = state.get("conversation_logger")
            if logger:
                logger.log_tester(
                    iteration=state.get("iteration", 0),
                    test_results=f"TIMEOUT: Execution exceeded {timeout_str}. Training took too long. Recommend reducing timesteps.",
                    execution_stdout="",
                    execution_stderr=f"Execution timeout after {execution_timeout} seconds",
                    execution_time=execution_timeout
                )

            return {
                "test_results": f"TIMEOUT: Execution exceeded {timeout_str}. Training took too long. Recommend reducing timesteps or simplifying the training loop. No metrics or videos were generated.",
                "execution_stdout": "",
                "execution_stderr": f"Execution timeout after {execution_timeout} seconds"
            }
        except Exception as e:
            print("\n" + "─" * 70)
            print(f"[bold red]💥 UNEXPECTED ERROR[/bold red]")
            print("─" * 70)
            print(f"[red]{type(e).__name__}: {e}[/red]")
            print(f"[dim]   Something went wrong outside the sandbox[/dim]")
            print("─" * 70 + "\n")
            
            # Log to conversation file
            logger = state.get("conversation_logger")
            if logger:
                logger.log_tester(
                    iteration=state.get("iteration", 0),
                    test_results=f"Unexpected: {str(e)[:200]}",
                    execution_stdout="",
                    execution_stderr=str(e),
                    execution_time=0
                )
            
            return {
                "test_results": f"Unexpected: {str(e)[:200]}",
                "execution_stdout": "",
                "execution_stderr": str(e)
            }