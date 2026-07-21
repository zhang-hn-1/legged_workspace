#!/usr/bin/env python3
"""Headless point-to-point control baseline for an Isaac Gym sphere."""

import math
import sys

from isaacgym import gymapi, gymtorch, gymutil
import torch


def main() -> int:
    custom_parameters = [
        {
            "name": "--num_envs",
            "type": int,
            "default": 4,
            "help": "Number of parallel environments",
        },
        {
            "name": "--settle_steps",
            "type": int,
            "default": 120,
            "help": "Initial settling steps",
        },
        {
            "name": "--control_steps",
            "type": int,
            "default": 900,
            "help": "Maximum point-to-point control steps",
        },
        {
            "name": "--target_distance",
            "type": float,
            "default": 1.5,
            "help": "Target distance from the initial position",
        },
        {
            "name": "--max_speed",
            "type": float,
            "default": 0.8,
            "help": "Maximum desired planar speed",
        },
        {
            "name": "--velocity_gain",
            "type": float,
            "default": 2.5,
            "help": "Planar velocity feedback gain",
        },
        {
            "name": "--max_torque",
            "type": float,
            "default": 1.5,
            "help": "Maximum rolling torque magnitude",
        },
        {
            "name": "--success_radius",
            "type": float,
            "default": 0.15,
            "help": "Distance threshold for reaching the target",
        },
    ]

    args = gymutil.parse_arguments(
        description="Sphere point-to-point controller",
        custom_parameters=custom_parameters,
    )

    if args.physics_engine != gymapi.SIM_PHYSX:
        raise RuntimeError("This test requires PhysX")

    if args.num_envs < 4:
        raise ValueError("--num_envs must be at least 4")

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
        print("SPHERE_TARGET_TEST=FAIL_CREATE_SIM")
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
            print(f"SPHERE_TARGET_TEST=FAIL_ACTOR_{env_id}")
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

    gym.prepare_sim(sim)

    state_descriptor = gym.acquire_rigid_body_state_tensor(sim)
    states = gymtorch.wrap_tensor(state_descriptor).view(-1, 13)

    if states.shape[0] != args.num_envs:
        print(
            "SPHERE_TARGET_TEST="
            f"FAIL_BODY_COUNT_EXPECTED_{args.num_envs}_ACTUAL_{states.shape[0]}"
        )
        gym.destroy_sim(sim)
        return 1

    device = states.device

    forces = torch.zeros(
        (args.num_envs, 3),
        dtype=torch.float32,
        device=device,
    )

    torques = torch.zeros_like(forces)

    # Initial physical settling.
    for _ in range(args.settle_steps):
        gym.apply_rigid_body_force_tensors(
            sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.refresh_rigid_body_state_tensor(sim)

    initial_position = states[:, 0:3].clone()

    # Four basic target directions: +X, -X, +Y, -Y.
    base_offsets = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    repeats = math.ceil(args.num_envs / 4)
    target_offsets = base_offsets.repeat(repeats, 1)
    target_offsets = target_offsets[: args.num_envs]
    target_offsets *= args.target_distance

    targets = initial_position[:, 0:2] + target_offsets

    success_step = torch.full(
        (args.num_envs,),
        -1,
        dtype=torch.long,
        device=device,
    )

    print(f"SIM_DEVICE={args.sim_device}")
    print(f"PIPELINE={'gpu' if args.use_gpu_pipeline else 'cpu'}")
    print(f"NUM_ENVS={args.num_envs}")
    print(f"CONTROL_STEPS={args.control_steps}")
    print(f"TARGET_DISTANCE={args.target_distance}")
    print(f"SUCCESS_RADIUS={args.success_radius}")

    for step in range(args.control_steps):
        gym.refresh_rigid_body_state_tensor(sim)

        position_xy = states[:, 0:2]
        velocity_xy = states[:, 7:9]

        target_error = targets - position_xy
        distance = torch.linalg.norm(
            target_error,
            dim=1,
        )

        newly_successful = (
            (distance <= args.success_radius)
            & (success_step < 0)
        )

        success_step[newly_successful] = step

        direction = target_error / distance.clamp_min(1.0e-6).unsqueeze(1)

        desired_speed = torch.clamp(
            distance,
            min=0.0,
            max=args.max_speed,
        )

        desired_velocity = direction * desired_speed.unsqueeze(1)

        planar_control = (
            args.velocity_gain
            * (desired_velocity - velocity_xy)
        )

        # Desired +X motion requires +Y rolling torque.
        # Desired +Y motion requires -X rolling torque.
        raw_torque_xy = torch.stack(
            (
                -planar_control[:, 1],
                planar_control[:, 0],
            ),
            dim=1,
        )

        torque_norm = torch.linalg.norm(
            raw_torque_xy,
            dim=1,
            keepdim=True,
        )

        torque_scale = torch.clamp(
            args.max_torque / torque_norm.clamp_min(1.0e-6),
            max=1.0,
        )

        torque_xy = raw_torque_xy * torque_scale

        active = success_step < 0
        torques.zero_()
        torques[active, 0:2] = torque_xy[active]

        gym.apply_rigid_body_force_tensors(
            sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if step == 0 or (step + 1) % 120 == 0:
            reached = int((success_step >= 0).sum().item())
            print(
                f"CONTROL_STEP={step + 1}/{args.control_steps} "
                f"REACHED={reached}/{args.num_envs}"
            )

        if bool((success_step >= 0).all().item()):
            print(f"ALL_TARGETS_REACHED_AT_STEP={step}")
            break

    gym.refresh_rigid_body_state_tensor(sim)

    final_position = states[:, 0:3].clone()
    final_velocity = states[:, 7:10].clone()

    final_distance = torch.linalg.norm(
        targets - final_position[:, 0:2],
        dim=1,
    )

    finite = bool(
        torch.isfinite(states).all().item()
    )

    print("===== TARGET RESULTS =====")

    for env_id in range(args.num_envs):
        target = targets[env_id].detach().cpu().tolist()
        position = final_position[env_id].detach().cpu().tolist()
        velocity = final_velocity[env_id].detach().cpu().tolist()
        distance = float(final_distance[env_id].item())
        reached_step = int(success_step[env_id].item())

        print(
            f"ENV={env_id} "
            f"TARGET=({target[0]:.4f},{target[1]:.4f}) "
            f"FINAL_POS=({position[0]:.4f},{position[1]:.4f},{position[2]:.4f}) "
            f"FINAL_VEL=({velocity[0]:.4f},{velocity[1]:.4f},{velocity[2]:.4f}) "
            f"FINAL_DISTANCE={distance:.4f} "
            f"SUCCESS_STEP={reached_step}"
        )

    all_reached = bool((success_step >= 0).all().item())

    print(f"FINITE_STATES={finite}")
    print(f"ALL_TARGETS_REACHED={all_reached}")
    print(
        "MAX_FINAL_DISTANCE="
        f"{float(final_distance.max().item()):.6f}"
    )

    gym.destroy_sim(sim)

    if finite and all_reached:
        print("SPHERE_TARGET_TEST=PASS")
        return 0

    print("SPHERE_TARGET_TEST=FAIL_BEHAVIOR")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            "SPHERE_TARGET_TEST="
            f"FAIL_EXCEPTION:{type(exc).__name__}:{exc}"
        )
        raise
