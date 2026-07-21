#!/usr/bin/env python3

import os
import sys
import platform
import traceback
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_VERSIONS = {
    "python": "3.8.20",
    "torch": "1.10.0+cu113",
    "cuda": "11.3",
    "numpy": "1.23.5",
    "setuptools": "59.5.0",
}

failures = []
warnings = []


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def record_failure(message):
    failures.append(message)
    print("[FAIL]", message)


def record_warning(message):
    warnings.append(message)
    print("[WARN]", message)


def record_pass(message):
    print("[PASS]", message)


section("1. Python 环境")

python_version = platform.python_version()

print("Workspace root :", WORKSPACE_ROOT)
print("Python executable:", sys.executable)
print("Python version   :", python_version)
print("Python compiler  :", platform.python_compiler())
print("Platform         :", platform.platform())

if python_version == EXPECTED_VERSIONS["python"]:
    record_pass("Python 版本符合预期")
else:
    record_warning(
        "Python 版本与记录不一致："
        f"expected={EXPECTED_VERSIONS['python']}, actual={python_version}"
    )


section("2. 工作区结构")

required_paths = [
    WORKSPACE_ROOT / "isaacgym",
    WORKSPACE_ROOT / "legged_gym",
    WORKSPACE_ROOT / "rsl_rl",
    WORKSPACE_ROOT / "sru_rotunbot_study",
    WORKSPACE_ROOT
    / "legged_gym"
    / "legged_gym"
    / "envs"
    / "anymal_c"
    / "flat"
    / "anymal_c_history.py",
    WORKSPACE_ROOT
    / "rsl_rl"
    / "rsl_rl"
    / "modules"
    / "actor_critic_sru_lh.py",
    WORKSPACE_ROOT
    / "sru_rotunbot_study"
    / "rotunbot_sru"
    / "memory.py",
]

for path in required_paths:
    if path.exists():
        record_pass(str(path.relative_to(WORKSPACE_ROOT)))
    else:
        record_failure(f"缺少路径：{path}")


section("3. Isaac Gym 导入")

try:
    # Isaac Gym 必须先于 PyTorch 导入。
    import isaacgym
    from isaacgym import gymapi, gymtorch

    print("isaacgym path:", getattr(isaacgym, "__file__", None))
    print("gymapi path  :", getattr(gymapi, "__file__", None))
    print("gymtorch path:", getattr(gymtorch, "__file__", None))
    record_pass("Isaac Gym、gymapi 和 gymtorch 导入成功")
except Exception as exc:
    record_failure(f"Isaac Gym 导入失败：{exc}")
    traceback.print_exc()


section("4. 核心 Python 依赖")

try:
    import numpy
    import setuptools
    import torch

    versions = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "setuptools": setuptools.__version__,
    }

    for name, actual in versions.items():
        expected = EXPECTED_VERSIONS[name]
        print(f"{name:12s}: {actual}")

        if str(actual) == expected:
            record_pass(f"{name} 版本符合预期")
        else:
            record_warning(
                f"{name} 版本与记录不一致：expected={expected}, actual={actual}"
            )

    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        record_failure("PyTorch 无法访问 CUDA")
    else:
        gpu_count = torch.cuda.device_count()
        print("Visible GPU count:", gpu_count)

        if gpu_count < 1:
            record_failure("没有可见 GPU")
        else:
            for index in range(gpu_count):
                properties = torch.cuda.get_device_properties(index)
                memory_gib = properties.total_memory / 1024**3

                print(
                    f"GPU {index}: {properties.name}, "
                    f"{memory_gib:.2f} GiB, "
                    f"compute capability "
                    f"{properties.major}.{properties.minor}"
                )

            record_pass("CUDA 和 GPU 检查通过")

except Exception as exc:
    record_failure(f"核心依赖检查失败：{exc}")
    traceback.print_exc()


section("5. SRU 与 rsl_rl 导入")

try:
    import rotunbot_sru
    from rotunbot_sru.memory import SRUMemoryEncoder
    from rsl_rl.modules import ActorCriticSRULH

    print(
        "rotunbot_sru:",
        getattr(rotunbot_sru, "__file__", None),
    )
    print(
        "SRUMemoryEncoder module:",
        SRUMemoryEncoder.__module__,
    )
    print(
        "ActorCriticSRULH module:",
        ActorCriticSRULH.__module__,
    )

    rotunbot_path = Path(rotunbot_sru.__file__).resolve()
    expected_rotunbot_root = (
        WORKSPACE_ROOT / "sru_rotunbot_study"
    ).resolve()

    if expected_rotunbot_root in rotunbot_path.parents:
        record_pass("rotunbot_sru 指向总工作区子模块")
    else:
        record_failure(
            "rotunbot_sru 导入路径错误："
            f"{rotunbot_path}"
        )

    record_pass("SRUMemoryEncoder 导入成功")
    record_pass("ActorCriticSRULH 导入成功")

except Exception as exc:
    record_failure(f"SRU 或 rsl_rl 导入失败：{exc}")
    traceback.print_exc()


section("6. 环境变量")

print("CUDA_VISIBLE_DEVICES:",
      os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
print("PYTHONPATH:",
      os.environ.get("PYTHONPATH", "<not set>"))


section("7. 最终结果")

print("Warnings:", len(warnings))
for warning in warnings:
    print("  -", warning)

print("Failures:", len(failures))
for failure in failures:
    print("  -", failure)

if failures:
    print()
    print("ENVIRONMENT_CHECK=FAIL")
    sys.exit(1)

print()
print("ENVIRONMENT_CHECK=PASS")
