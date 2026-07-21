#!/usr/bin/env python3
"""Vectorized sphere navigation environment smoke test."""

import math
import sys

from isaacgym import gymapi, gymtorch, gymutil
import torch


def main() -> int:
    custom_parameters = [
        {
            "name": "--num_envs",
            "type": int,
            "default": 64,
            "help": "Number of parallel environments",
        },
        {
            "name": "--steps",
            "type": int,
            "default": 3000,
            "help": "Total environment steps",
        },
        {
            "name": "--episode_length",
            "type": int,
            "default": 360,
            "help": "Base episode length",
        },
        {
            "name": "--action_scale",
            "type": float,
            "default": 1.5,
            "help": "Maximum rolling torque",
        },
        {
            "name": "--success_radius",
            "type": float,
            "default": 0.15,
            "help": "Target success distance",
        },
        {
            "name": "--success_speed",
            "type": float,
            "default": 0.30,
            "help": "Maximum speed at successful arrival",
        },
        {
            "name": "--dwell_steps",
            "type": int,
            "default": 5,
            "help": "Consecutive stable steps required",
        },
    ]

    args = gymutil.parse_arguments(
        description="Sphere RL environment smoke test",
        custom_parameters=custom_parameters,
    )

    if args.physics_engine != gymapi.SIM_PHYSX:
        raise RuntimeError("This test requires PhysX")

    if args.num_envs < 4:
        raise ValueError("--num_envs must be at least 4")

    torch.manual_seed(1)

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.num_subscenes = args.subscenes
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    sim = gym.create_sim(
        args.compute_device_id,
        -1,
        args.physics_engine,
        sim_params,
    )

    if sim is None:
        print("SPHERE_RL_ENV_SMOKE=FAIL_CREATE_SIM")
        return 1

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane.static_friction = 1.0
    plane.dynamic_friction = 0.9
    plane.restitution = 0.0
    gym.add_ground(sim, plane)

    radius = 0.20

    asset_options = gymapi.AssetOptions()
    asset_options.density = 500.0
    asset_options.linear_damping = 0.08
    asset_options.angular_damping = 0.08

    sphere_asset = gym.create_sphere(
        sim,
        radius,
        asset_options,
    )

    env_lower = gymapi.Vec3(-3.0, -3.0, 0.0)
    env_upper = gymapi.Vec3(3.0, 3.0, 2.0)
    envs_per_row = int(math.ceil(math.sqrt(args.num_envs)))

    actor_indices_cpu = []

    for env_id in range(args.num_envs):
        env = gym.create_env(
            sim,
            env_lower,
            env_upper,
            envs_per_row,
        )

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(
            0.0,
            0.0,
            radius + 0.02,
        )

        actor = gym.create_actor(
            env,
            sphere_asset,
            pose,
            f"sphere_{env_id}",
            env_id,
            0,
        )

        if actor < 0:
            gym.destroy_sim(sim)
            print(f"SPHERE_RL_ENV_SMOKE=FAIL_ACTOR_{env_id}")
            return 1

        shape_props = gym.get_actor_rigid_shape_properties(
            env,
            actor,
        )

        for prop in shape_props:
            prop.friction = 1.0
            prop.rolling_friction = 0.0
            prop.torsion_friction = 0.0
            prop.restitution = 0.0

        gym.set_actor_rigid_shape_properties(
            env,
            actor,
            shape_props,
        )

        actor_index = gym.get_actor_index(
            env,
            actor,
            gymapi.DOMAIN_SIM,
        )
        actor_indices_cpu.append(actor_index)

    gym.prepare_sim(sim)

    root_descriptor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_descriptor).view(-1, 13)

    device = root_states.device

    actor_indices = torch.tensor(
        actor_indices_cpu,
        dtype=torch.int32,
        device=device,
    )

    actor_rows = actor_indices.long()

    if actor_rows.numel() != args.num_envs:
        gym.destroy_sim(sim)
        print("SPHERE_RL_ENV_SMOKE=FAIL_ACTOR_INDEX_COUNT")
        return 1

    forces = torch.zeros(
        (args.num_envs, 3),
        dtype=torch.float32,
        device=device,
    )
    torques = torch.zeros_like(forces)

    targets = torch.zeros(
        (args.num_envs, 2),
        dtype=torch.float32,
        device=device,
    )

    episode_length = torch.zeros(
        args.num_envs,
        dtype=torch.long,
        device=device,
    )

    episode_limits = (
        args.episode_length
        + (torch.arange(args.num_envs, device=device) % 8) * 20
    )

    dwell_buffer = torch.zeros_like(episode_length)
    reset_count = torch.zeros_like(episode_length)
    success_count = torch.zeros_like(episode_length)

    previous_distance = torch.zeros(
        args.num_envs,
        dtype=torch.float32,
        device=device,
    )

    cumulative_reward = torch.zeros(
        args.num_envs,
        dtype=torch.float32,
        device=device,
    )

    partial_reset_seen = False
    maximum_reset_batch = 0
    total_reset_events = 0

    def reset_envs(env_ids: torch.Tensor) -> None:
        nonlocal total_reset_events

        if env_ids.numel() == 0:
            return

        rows = actor_rows[env_ids]

        root_states[rows, :] = 0.0
        root_states[rows, 2] = radius + 0.002
        root_states[rows, 6] = 1.0

        selected_indices = actor_indices[env_ids].contiguous()

        success = gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(selected_indices),
            selected_indices.numel(),
        )

        if not success:
            raise RuntimeError("set_actor_root_state_tensor_indexed failed")

        env_float = env_ids.float()
        reset_float = reset_count[env_ids].float()

        angle = torch.remainder(
            env_float * 1.6180339 + reset_float * 0.731,
            2.0 * math.pi,
        )

        target_radius = (
            0.8
            + (env_ids % 7).float() * 0.18
            + (reset_count[env_ids] % 3).float() * 0.05
        )

        targets[env_ids, 0] = torch.cos(angle) * target_radius
        targets[env_ids, 1] = torch.sin(angle) * target_radius

        previous_distance[env_ids] = target_radius
        episode_length[env_ids] = 0
        dwell_buffer[env_ids] = 0
        cumulative_reward[env_ids] = 0.0
        reset_count[env_ids] += 1

        total_reset_events += int(env_ids.numel())

    all_env_ids = torch.arange(
        args.num_envs,
        dtype=torch.long,
        device=device,
    )

    reset_envs(all_env_ids)

    for _ in range(60):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.refresh_actor_root_state_tensor(sim)

    print(f"SIM_DEVICE={args.sim_device}")
    print(f"PIPELINE={'gpu' if args.use_gpu_pipeline else 'cpu'}")
    print(f"NUM_ENVS={args.num_envs}")
    print("OBSERVATION_DIM=6")
    print("ACTION_DIM=2")
    print(f"TOTAL_STEPS={args.steps}")

    reward_finite = True
    observation_finite = True

    for global_step in range(args.steps):
        gym.refresh_actor_root_state_tensor(sim)

        states = root_states[actor_rows]

        position_xy = states[:, 0:2]
        velocity_xy = states[:, 7:9]
        angular_velocity_xy = states[:, 10:12]

        target_error = targets - position_xy
        distance = torch.linalg.norm(
            target_error,
            dim=1,
        )

        observations = torch.cat(
            (
                target_error,
                velocity_xy,
                angular_velocity_xy,
            ),
            dim=1,
        )

        observation_finite = (
            observation_finite
            and bool(torch.isfinite(observations).all().item())
        )

        direction = target_error / distance.clamp_min(
            1.0e-6
        ).unsqueeze(1)

        desired_speed = torch.clamp(
            distance * 1.2,
            min=0.0,
            max=0.8,
        )

        desired_velocity = direction * desired_speed.unsqueeze(1)

        planar_command = 2.5 * (
            desired_velocity - velocity_xy
        )

        action = torch.clamp(
            planar_command / args.action_scale,
            min=-1.0,
            max=1.0,
        )

        torques.zero_()
        torques[:, 0] = -action[:, 1] * args.action_scale
        torques[:, 1] = action[:, 0] * args.action_scale

        gym.apply_rigid_body_force_tensors(
            sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)

        new_states = root_states[actor_rows]
        new_position_xy = new_states[:, 0:2]
        new_velocity_xy = new_states[:, 7:9]

        new_distance = torch.linalg.norm(
            targets - new_position_xy,
            dim=1,
        )
        planar_speed = torch.linalg.norm(
            new_velocity_xy,
            dim=1,
        )

        progress_reward = previous_distance - new_distance
        action_penalty = 0.002 * torch.sum(
            action * action,
            dim=1,
        )

        stable_in_goal = (
            (new_distance <= args.success_radius)
            & (planar_speed <= args.success_speed)
        )

        dwell_buffer = torch.where(
            stable_in_goal,
            dwell_buffer + 1,
            torch.zeros_like(dwell_buffer),
        )

        successful = dwell_buffer >= args.dwell_steps

        reward = (
            10.0 * progress_reward
            - action_penalty
            + successful.float() * 5.0
        )

        reward_finite = (
            reward_finite
            and bool(torch.isfinite(reward).all().item())
        )

        cumulative_reward += reward
        previous_distance = new_distance
        episode_length += 1

        timed_out = episode_length >= episode_limits
        done = successful | timed_out

        if bool(done.any().item()):
            done_ids = torch.nonzero(
                done,
                as_tuple=False,
            ).flatten()

            done_count = int(done_ids.numel())
            maximum_reset_batch = max(
                maximum_reset_batch,
                done_count,
            )

            if 0 < done_count < args.num_envs:
                partial_reset_seen = True

            success_count[done_ids] += successful[done_ids].long()
            reset_envs(done_ids)

        if global_step == 0 or (global_step + 1) % 250 == 0:
            print(
                f"STEP={global_step + 1}/{args.steps} "
                f"RESETS={total_reset_events} "
                f"SUCCESSES={int(success_count.sum().item())} "
                f"MEAN_DISTANCE={float(new_distance.mean().item()):.4f}"
            )

    gym.refresh_actor_root_state_tensor(sim)

    final_states = root_states[actor_rows]

    state_finite = bool(
        torch.isfinite(final_states).all().item()
    )

    total_successes = int(success_count.sum().item())

    print("===== RL ENVIRONMENT SUMMARY =====")
    print(f"STATE_FINITE={state_finite}")
    print(f"OBSERVATION_FINITE={observation_finite}")
    print(f"REWARD_FINITE={reward_finite}")
    print(f"TOTAL_RESET_EVENTS={total_reset_events}")
    print(f"TOTAL_SUCCESSES={total_successes}")
    print(f"MAXIMUM_RESET_BATCH={maximum_reset_batch}")
    print(f"PARTIAL_RESET_SEEN={partial_reset_seen}")
    print(
        "MEAN_RESETS_PER_ENV="
        f"{float(reset_count.float().mean().item()):.4f}"
    )
    print(
        "MIN_RESETS_PER_ENV="
        f"{int(reset_count.min().item())}"
    )
    print(
        "MAX_RESETS_PER_ENV="
        f"{int(reset_count.max().item())}"
    )

    passed = (
        state_finite
        and observation_finite
        and reward_finite
        and total_reset_events > args.num_envs
        and total_successes > 0
        and partial_reset_seen
    )

    gym.destroy_sim(sim)

    if passed:
        print("SPHERE_RL_ENV_SMOKE=PASS")
        return 0

    print("SPHERE_RL_ENV_SMOKE=FAIL_BEHAVIOR")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            "SPHERE_RL_ENV_SMOKE="
            f"FAIL_EXCEPTION:{type(exc).__name__}:{exc}"
        )
        raise
