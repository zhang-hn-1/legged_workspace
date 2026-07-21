#!/usr/bin/env python3
"""Headless controlled-sphere test using Isaac Gym tensor APIs."""

import math
import sys

from isaacgym import gymapi, gymtorch, gymutil
import torch


def mode_name(env_id: int) -> str:
    mode = env_id % 4
    if mode == 0:
        return "zero"
    if mode == 1:
        return "forward_y_positive"
    if mode == 2:
        return "backward_y_negative"
    return "yaw_z_positive"


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
            "help": "Zero-action settling steps",
        },
        {
            "name": "--drive_steps",
            "type": int,
            "default": 240,
            "help": "Controlled torque steps",
        },
        {
            "name": "--roll_torque",
            "type": float,
            "default": 1.5,
            "help": "Torque about environment Y axis",
        },
        {
            "name": "--yaw_torque",
            "type": float,
            "default": 0.6,
            "help": "Torque about environment Z axis",
        },
    ]

    args = gymutil.parse_arguments(
        description="Controlled headless sphere test",
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

    # No viewer or graphics context.
    sim = gym.create_sim(
        args.compute_device_id,
        -1,
        args.physics_engine,
        sim_params,
    )

    if sim is None:
        print("CONTROLLED_SPHERE_TEST=FAIL_CREATE_SIM")
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
    asset_options.linear_damping = 0.05
    asset_options.angular_damping = 0.05

    sphere_asset = gym.create_sphere(sim, radius, asset_options)

    if sphere_asset is None:
        gym.destroy_sim(sim)
        print("CONTROLLED_SPHERE_TEST=FAIL_CREATE_ASSET")
        return 1

    env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
    env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
    envs_per_row = int(math.ceil(math.sqrt(args.num_envs)))

    actor_handles = []

    for env_id in range(args.num_envs):
        env = gym.create_env(
            sim,
            env_lower,
            env_upper,
            envs_per_row,
        )

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, radius + 0.02)

        actor = gym.create_actor(
            env,
            sphere_asset,
            pose,
            f"controlled_sphere_{env_id}",
            env_id,
            0,
        )

        if actor < 0:
            gym.destroy_sim(sim)
            print(f"CONTROLLED_SPHERE_TEST=FAIL_ACTOR_{env_id}")
            return 1

        shape_props = gym.get_actor_rigid_shape_properties(env, actor)

        for prop in shape_props:
            prop.friction = 1.0
            prop.rolling_friction = 0.0
            prop.torsion_friction = 0.0
            prop.restitution = 0.0

        gym.set_actor_rigid_shape_properties(env, actor, shape_props)
        actor_handles.append(actor)

    gym.prepare_sim(sim)

    state_descriptor = gym.acquire_rigid_body_state_tensor(sim)
    rigid_body_states = gymtorch.wrap_tensor(state_descriptor).view(-1, 13)

    num_bodies = rigid_body_states.shape[0]

    if num_bodies != args.num_envs:
        gym.destroy_sim(sim)
        print(
            "CONTROLLED_SPHERE_TEST="
            f"FAIL_BODY_COUNT_EXPECTED_{args.num_envs}_ACTUAL_{num_bodies}"
        )
        return 1

    tensor_device = rigid_body_states.device

    forces = torch.zeros(
        (num_bodies, 3),
        dtype=torch.float32,
        device=tensor_device,
    )

    torques = torch.zeros_like(forces)

    print(f"SIM_DEVICE={args.sim_device}")
    print(f"PIPELINE={'gpu' if args.use_gpu_pipeline else 'cpu'}")
    print(f"NUM_ENVS={args.num_envs}")
    print(f"NUM_BODIES={num_bodies}")
    print(f"TENSOR_DEVICE={tensor_device}")
    print(f"SETTLE_STEPS={args.settle_steps}")
    print(f"DRIVE_STEPS={args.drive_steps}")

    # Let all spheres settle without control.
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
    initial_states = rigid_body_states.clone()

    # Configure one control mode per environment.
    for env_id in range(args.num_envs):
        mode = env_id % 4

        if mode == 1:
            torques[env_id, 1] = args.roll_torque
        elif mode == 2:
            torques[env_id, 1] = -args.roll_torque
        elif mode == 3:
            torques[env_id, 2] = args.yaw_torque

    for step in range(args.drive_steps):
        gym.apply_rigid_body_force_tensors(
            sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if step == 0 or (step + 1) % 60 == 0:
            gym.refresh_rigid_body_state_tensor(sim)
            print(f"DRIVE_STEP={step + 1}/{args.drive_steps}")

    gym.refresh_rigid_body_state_tensor(sim)
    final_states = rigid_body_states.clone()

    finite = bool(torch.isfinite(final_states).all().item())

    delta_position = final_states[:, 0:3] - initial_states[:, 0:3]
    final_linear_velocity = final_states[:, 7:10]
    final_angular_velocity = final_states[:, 10:13]

    print("===== FINAL STATES =====")

    for env_id in range(args.num_envs):
        dp = delta_position[env_id].detach().cpu().tolist()
        lv = final_linear_velocity[env_id].detach().cpu().tolist()
        av = final_angular_velocity[env_id].detach().cpu().tolist()

        print(
            f"ENV={env_id} "
            f"MODE={mode_name(env_id)} "
            f"DELTA_POS=({dp[0]:.6f},{dp[1]:.6f},{dp[2]:.6f}) "
            f"LIN_VEL=({lv[0]:.6f},{lv[1]:.6f},{lv[2]:.6f}) "
            f"ANG_VEL=({av[0]:.6f},{av[1]:.6f},{av[2]:.6f})"
        )

    forward_dx = float(delta_position[1, 0].item())
    backward_dx = float(delta_position[2, 0].item())
    yaw_wz = float(final_angular_velocity[3, 2].item())
    zero_distance = float(torch.linalg.norm(delta_position[0, 0:2]).item())

    opposite_roll_directions = forward_dx * backward_dx < 0.0
    roll_motion_detected = abs(forward_dx) > 0.02 and abs(backward_dx) > 0.02
    yaw_motion_detected = abs(yaw_wz) > 0.05
    zero_action_stable = zero_distance < 0.10

    print(f"FINITE_STATES={finite}")
    print(f"ZERO_ACTION_DISTANCE={zero_distance:.6f}")
    print(f"FORWARD_DX={forward_dx:.6f}")
    print(f"BACKWARD_DX={backward_dx:.6f}")
    print(f"YAW_ANGULAR_VELOCITY_Z={yaw_wz:.6f}")
    print(f"ZERO_ACTION_STABLE={zero_action_stable}")
    print(f"ROLL_MOTION_DETECTED={roll_motion_detected}")
    print(f"OPPOSITE_ROLL_DIRECTIONS={opposite_roll_directions}")
    print(f"YAW_MOTION_DETECTED={yaw_motion_detected}")

    passed = (
        finite
        and zero_action_stable
        and roll_motion_detected
        and opposite_roll_directions
        and yaw_motion_detected
    )

    gym.destroy_sim(sim)

    if passed:
        print("CONTROLLED_SPHERE_TEST=PASS")
        return 0

    print("CONTROLLED_SPHERE_TEST=FAIL_BEHAVIOR")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            "CONTROLLED_SPHERE_TEST="
            f"FAIL_EXCEPTION:{type(exc).__name__}:{exc}"
        )
        raise
