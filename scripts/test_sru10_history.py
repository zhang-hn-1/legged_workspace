#!/usr/bin/env python3
"""Integration tests for the ANYmal ten-frame observation history."""

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Make the test runnable without relying on an externally configured PYTHONPATH.
for path in (
    WORKSPACE_ROOT / "isaacgym" / "python",
    WORKSPACE_ROOT / "legged_gym",
    WORKSPACE_ROOT / "rsl_rl",
    WORKSPACE_ROOT / "sru_rotunbot_study",
):
    sys.path.insert(0, str(path))


# Isaac Gym must be imported before PyTorch.
from isaacgym import gymapi, gymtorch  # noqa: E402,F401

import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402


TASK_NAME = "anymal_c_flat_sru10"
EXPECTED_HISTORY_STEPS = 10
EXPECTED_SINGLE_OBS = 48
EXPECTED_ACTOR_OBS = 480
EXPECTED_CRITIC_OBS = 48


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def passed(message):
    print("[PASS]", message)


def assert_true(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def assert_shape(tensor, expected_shape, name):
    actual_shape = tuple(tensor.shape)

    if actual_shape != tuple(expected_shape):
        raise AssertionError(
            f"{name} shape mismatch: "
            f"expected={tuple(expected_shape)}, actual={actual_shape}"
        )

    passed(f"{name} shape = {actual_shape}")


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        invalid_count = int((~torch.isfinite(tensor)).sum().item())
        raise AssertionError(
            f"{name} contains {invalid_count} NaN or Inf values"
        )

    passed(f"{name} contains only finite values")


def assert_close(actual, expected, name, atol=1e-6, rtol=1e-5):
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{name} shape mismatch: "
            f"actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )

    if actual.numel() == 0:
        passed(f"{name}: empty selection skipped")
        return

    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_error = float(torch.max(torch.abs(actual - expected)).item())
        raise AssertionError(
            f"{name} mismatch: max_abs_error={max_error:.8e}"
        )

    passed(name)


def check_observation_views(env, label):
    """Verify actor and critic views against the internal history buffer."""

    history = env.history_buf
    actor = env.obs_buf
    critic = env.privileged_obs_buf

    assert_true(
        critic is not None,
        f"{label}: privileged observation buffer is None",
    )

    clip_value = float(env.cfg.normalization.clip_observations)

    expected_actor = torch.clamp(
        history.reshape(env.num_envs, -1),
        -clip_value,
        clip_value,
    )

    expected_critic = torch.clamp(
        history[:, -1],
        -clip_value,
        clip_value,
    )

    assert_close(
        actor,
        expected_actor,
        f"{label}: actor observation equals flattened history",
    )

    assert_close(
        critic,
        expected_critic,
        f"{label}: critic observation equals latest frame",
    )

    assert_finite(actor, f"{label}: actor observation")
    assert_finite(critic, f"{label}: critic observation")
    assert_finite(history, f"{label}: history buffer")


def check_repeated_history(history, mask, label):
    """Verify selected environments contain ten copies of the latest frame."""

    selected = history[mask]

    if selected.shape[0] == 0:
        passed(f"{label}: no selected environments")
        return

    expected = selected[:, -1:].expand_as(selected)

    assert_close(
        selected,
        expected,
        label,
    )


def main():
    args = get_args()

    section("1. Build test environment")

    if args.task != TASK_NAME:
        print(
            f"[INFO] overriding task {args.task!r} with {TASK_NAME!r}"
        )
        args.task = TASK_NAME

    # Keep this integration test small and predictable.
    if args.num_envs is None:
        args.num_envs = 4

    assert_true(
        args.num_envs >= 4,
        "At least four environments are required",
    )

    env_cfg, _ = task_registry.get_cfgs(TASK_NAME)

    # Remove observation noise so equality checks are deterministic.
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(
        name=TASK_NAME,
        args=args,
        env_cfg=env_cfg,
    )

    print("Task             :", TASK_NAME)
    print("Environment count:", env.num_envs)
    print("Device           :", env.device)
    print("History steps    :", env.history_steps)
    print("Single obs size  :", env.single_observation_size)
    print("Actor obs size   :", env.num_obs)
    print("Critic obs size  :", env.num_privileged_obs)

    section("2. Shape and configuration checks")

    assert_true(
        env.history_steps == EXPECTED_HISTORY_STEPS,
        "history_steps must equal 10",
    )
    passed("history_steps = 10")

    assert_true(
        env.single_observation_size == EXPECTED_SINGLE_OBS,
        "single_observation_size must equal 48",
    )
    passed("single_observation_size = 48")

    assert_true(
        env.num_obs == EXPECTED_ACTOR_OBS,
        "actor observation size must equal 480",
    )
    passed("actor observation size = 480")

    assert_true(
        env.num_privileged_obs == EXPECTED_CRITIC_OBS,
        "critic observation size must equal 48",
    )
    passed("critic observation size = 48")

    assert_shape(
        env.history_buf,
        (
            env.num_envs,
            EXPECTED_HISTORY_STEPS,
            EXPECTED_SINGLE_OBS,
        ),
        "history_buf",
    )

    assert_shape(
        env.obs_buf,
        (env.num_envs, EXPECTED_ACTOR_OBS),
        "obs_buf",
    )

    assert_shape(
        env.privileged_obs_buf,
        (env.num_envs, EXPECTED_CRITIC_OBS),
        "privileged_obs_buf",
    )

    section("3. Initial history semantics")

    assert_true(
        bool(env.history_initialized.all().item()),
        "All environments must be initialized after construction",
    )
    passed("all history_initialized flags are true")

    all_env_mask = torch.ones(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )

    check_repeated_history(
        env.history_buf,
        all_env_mask,
        "initial history contains ten copies of current frame",
    )

    check_observation_views(env, "initial state")

    section("4. Normal history shift")

    actions = torch.zeros(
        env.num_envs,
        env.num_actions,
        dtype=torch.float,
        device=env.device,
    )

    for step_index in range(3):
        before = env.history_buf.clone()

        _, _, _, dones, _ = env.step(actions)

        after = env.history_buf
        reset_mask = dones.bool()
        active_mask = ~reset_mask

        if bool(active_mask.any().item()):
            assert_close(
                after[active_mask, :-1],
                before[active_mask, 1:],
                f"step {step_index + 1}: active histories shift left",
            )

        if bool(reset_mask.any().item()):
            check_repeated_history(
                after,
                reset_mask,
                f"step {step_index + 1}: reset histories are reinitialized",
            )
        else:
            passed(
                f"step {step_index + 1}: no spontaneous environment reset"
            )

        check_observation_views(
            env,
            f"step {step_index + 1}",
        )

    section("5. Targeted reset semantics")

    assert_true(
        bool(env.history_initialized.all().item()),
        "All histories must be initialized before targeted reset",
    )

    reset_ids = torch.tensor(
        [0, 2],
        dtype=torch.long,
        device=env.device,
    )

    target_mask = torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )
    target_mask[reset_ids] = True
    unaffected_mask = ~target_mask

    before_reset = env.history_buf.clone()

    env.reset_idx(reset_ids)

    assert_true(
        not bool(env.history_initialized[target_mask].any().item()),
        "Only reset environments must be marked uninitialized",
    )
    passed("target environments marked uninitialized")

    assert_true(
        bool(env.history_initialized[unaffected_mask].all().item()),
        "Unaffected environments must remain initialized",
    )
    passed("unaffected environments remain initialized")

    assert_close(
        env.history_buf,
        before_reset,
        "reset_idx does not directly overwrite history data",
    )

    env.compute_observations()

    assert_true(
        bool(env.history_initialized.all().item()),
        "compute_observations must initialize reset histories",
    )
    passed("all histories initialized after observation computation")

    check_repeated_history(
        env.history_buf,
        target_mask,
        "target histories refilled with current frame",
    )

    assert_close(
        env.history_buf[unaffected_mask, :-1],
        before_reset[unaffected_mask, 1:],
        "unaffected histories continue normal left shift",
    )

    check_observation_views(env, "after targeted reset")

    section("6. Final result")

    print("SRU10_HISTORY_INITIALIZATION=PASS")
    print("SRU10_HISTORY_SHIFT=PASS")
    print("SRU10_TARGETED_RESET=PASS")
    print("SRU10_OBSERVATION_LAYOUT=PASS")
    print()
    print("SRU10_HISTORY_TEST=PASS")


if __name__ == "__main__":
    main()
