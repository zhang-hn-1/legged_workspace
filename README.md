# legged_workspace

Fixed-window SRU policy integration for Isaac Gym, legged_gym and rsl_rl.

## Components

- `legged_gym`: ANYmal environment and fixed-history observation integration
- `rsl_rl`: SRU actor-critic adapter and runner registration
- `sru_rotunbot_study`: SRU memory encoder implementation

## Validated environment

- Python 3.8.20
- PyTorch 1.10.0+cu113
- NumPy 1.23.5
- setuptools 59.5.0
- Isaac Gym Preview 4
- rsl_rl v1.0.2
- Tesla V100-SXM2-32GB

## Validated tasks

- Original `anymal_c_flat` PPO smoke test
- Single-frame `anymal_c_flat_sru`
- Ten-frame `anymal_c_flat_sru10`
- 64 parallel environments
- GPU PhysX and GPU pipeline
- Five PPO updates without NaN, shape mismatch or CUDA errors

## Clone

```bash
git clone --recurse-submodules git@github.com:zhang-hn-1/legged_workspace.git

Isaac Gym and the Python environment must be installed separately.
