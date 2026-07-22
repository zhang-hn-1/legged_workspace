#!/usr/bin/env python3
"""Minimal PPO training smoke test for a torque-controlled sphere."""

import math
import os
import sys
import time

from isaacgym import gymapi, gymtorch, gymutil
import torch

from legged_gym.envs.base.base_task import BaseTask
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class SphereNavigationCfg:
    class env:
        num_envs = 64
        num_observations = 6
        num_privileged_obs = None
        num_actions = 2


class SphereNavigationEnv(BaseTask):
    """Vectorized point-to-point sphere navigation environment."""

    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
        episode_length=360,
        action_scale=1.5,
        success_radius=0.15,
        success_speed=0.30,
        dwell_steps=5,
    ):
        self.cfg = cfg

        self.radius = 0.20
        self.action_scale = float(action_scale)
        self.success_radius = float(success_radius)
        self.success_speed = float(success_speed)
        self.dwell_steps = int(dwell_steps)
        self.max_episode_length = int(episode_length)

        self.envs = []
        self.actor_handles = []
        self.actor_indices_cpu = []

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            sim_device=sim_device,
            headless=headless,
        )

        root_descriptor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root_descriptor).view(-1, 13)

        self.actor_indices = torch.tensor(
            self.actor_indices_cpu,
            dtype=torch.int32,
            device=self.device,
        )
        self.actor_rows = self.actor_indices.long()

        if self.actor_rows.numel() != self.num_envs:
            raise RuntimeError(
                "Actor count mismatch: expected {}, got {}".format(
                    self.num_envs,
                    self.actor_rows.numel(),
                )
            )

        self.forces = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self.torques = torch.zeros_like(self.forces)

        self.targets = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float32,
            device=self.device,
        )

        self.previous_distance = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        self.dwell_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.success_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.reset_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = 1.0
        plane.dynamic_friction = 0.9
        plane.restitution = 0.0
        self.gym.add_ground(self.sim, plane)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 500.0
        asset_options.linear_damping = 0.08
        asset_options.angular_damping = 0.08

        sphere_asset = self.gym.create_sphere(
            self.sim,
            self.radius,
            asset_options,
        )

        if sphere_asset is None:
            raise RuntimeError("Failed to create sphere asset")

        env_lower = gymapi.Vec3(-3.0, -3.0, 0.0)
        env_upper = gymapi.Vec3(3.0, 3.0, 2.0)
        envs_per_row = int(math.ceil(math.sqrt(self.num_envs)))

        for env_id in range(self.num_envs):
            env = self.gym.create_env(
                self.sim,
                env_lower,
                env_upper,
                envs_per_row,
            )

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                0.0,
                0.0,
                self.radius + 0.002,
            )

            actor = self.gym.create_actor(
                env,
                sphere_asset,
                pose,
                "sphere_{}".format(env_id),
                env_id,
                0,
            )

            if actor < 0:
                raise RuntimeError(
                    "Failed to create actor for environment {}".format(env_id)
                )

            shape_props = self.gym.get_actor_rigid_shape_properties(
                env,
                actor,
            )

            for prop in shape_props:
                prop.friction = 1.0
                prop.rolling_friction = 0.0
                prop.torsion_friction = 0.0
                prop.restitution = 0.0

            self.gym.set_actor_rigid_shape_properties(
                env,
                actor,
                shape_props,
            )

            actor_index = self.gym.get_actor_index(
                env,
                actor,
                gymapi.DOMAIN_SIM,
            )

            self.envs.append(env)
            self.actor_handles.append(actor)
            self.actor_indices_cpu.append(actor_index)

        return self.sim

    def reset_idx(self, env_ids):
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.long()
        rows = self.actor_rows[env_ids]

        self.root_states[rows, :] = 0.0
        self.root_states[rows, 2] = self.radius + 0.002
        self.root_states[rows, 6] = 1.0

        selected_indices = self.actor_indices[env_ids].contiguous()

        result = self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(selected_indices),
            selected_indices.numel(),
        )

        if not result:
            raise RuntimeError(
                "set_actor_root_state_tensor_indexed failed"
            )

        count = env_ids.numel()

        angle = (
            2.0
            * math.pi
            * torch.rand(
                count,
                dtype=torch.float32,
                device=self.device,
            )
        )

        target_radius = (
            0.8
            + 1.2
            * torch.rand(
                count,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.targets[env_ids, 0] = torch.cos(angle) * target_radius
        self.targets[env_ids, 1] = torch.sin(angle) * target_radius

        self.previous_distance[env_ids] = target_radius
        self.episode_length_buf[env_ids] = 0
        self.dwell_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False

        self.forces[env_ids] = 0.0
        self.torques[env_ids] = 0.0

        self.reset_counter[env_ids] += 1

    def _compute_observations(self):
        states = self.root_states[self.actor_rows]

        position_xy = states[:, 0:2]
        velocity_xy = states[:, 7:9]
        angular_velocity_xy = states[:, 10:12]

        target_error = self.targets - position_xy

        # Basic manual normalization.
        self.obs_buf[:, 0:2] = target_error / 2.0
        self.obs_buf[:, 2:4] = velocity_xy / 2.0
        self.obs_buf[:, 4:6] = angular_velocity_xy / 10.0

    def step(self, actions):
        actions = torch.clamp(
            actions.to(self.device),
            min=-1.0,
            max=1.0,
        )

        self.torques.zero_()

        # Desired +X motion corresponds to +Y rolling torque.
        # Desired +Y motion corresponds to -X rolling torque.
        self.torques[:, 0] = (
            -actions[:, 1] * self.action_scale
        )
        self.torques[:, 1] = (
            actions[:, 0] * self.action_scale
        )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.ENV_SPACE,
        )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.episode_length_buf += 1

        states = self.root_states[self.actor_rows]

        position_xy = states[:, 0:2]
        velocity_xy = states[:, 7:9]

        target_error = self.targets - position_xy

        distance = torch.linalg.norm(
            target_error,
            dim=1,
        )

        planar_speed = torch.linalg.norm(
            velocity_xy,
            dim=1,
        )

        progress = self.previous_distance - distance

        stable_in_goal = (
            (distance <= self.success_radius)
            & (planar_speed <= self.success_speed)
        )

        self.dwell_buf = torch.where(
            stable_in_goal,
            self.dwell_buf + 1,
            torch.zeros_like(self.dwell_buf),
        )

        success = self.dwell_buf >= self.dwell_steps

        self.time_out_buf = (
            self.episode_length_buf >= self.max_episode_length
        )

        finite_state = torch.isfinite(states).all(dim=1)

        out_of_bounds = (
            (torch.abs(position_xy[:, 0]) > 5.0)
            | (torch.abs(position_xy[:, 1]) > 5.0)
            | (states[:, 2] < 0.0)
            | (states[:, 2] > 1.0)
        )

        invalid = (~finite_state) | out_of_bounds

        action_penalty = 0.002 * torch.sum(
            actions * actions,
            dim=1,
        )

        speed_penalty = 0.002 * planar_speed

        self.rew_buf[:] = (
            10.0 * progress
            - action_penalty
            - speed_penalty
            + 5.0 * success.float()
            - 2.0 * invalid.float()
        )

        self.previous_distance[:] = distance

        dones_bool = success | self.time_out_buf | invalid
        dones = dones_bool.long()

        self.reset_buf[:] = dones

        timeout_info = self.time_out_buf.clone()

        self.success_counter += success.long()

        done_ids = torch.nonzero(
            dones_bool,
            as_tuple=False,
        ).flatten()

        if done_ids.numel() > 0:
            self.reset_idx(done_ids)

        self._compute_observations()

        self.extras = {
            "time_outs": timeout_info,
        }

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            dones,
            self.extras,
        )


def build_train_cfg():
    return {
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 5,
            "save_interval": 50,
            "experiment_name": "sphere_ppo_smoke",
            "run_name": "mlp_baseline",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
    }


def main():
    custom_parameters = [
        {
            "name": "--num_envs",
            "type": int,
            "default": 64,
            "help": "Number of parallel environments",
        },
        {
            "name": "--iterations",
            "type": int,
            "default": 5,
            "help": "Number of PPO learning iterations",
        },
        {
            "name": "--episode_length",
            "type": int,
            "default": 360,
            "help": "Maximum episode length",
        },
        {
            "name": "--seed",
            "type": int,
            "default": 1,
            "help": "Random seed",
        },
    ]

    args = gymutil.parse_arguments(
        description="Sphere PPO smoke test",
        custom_parameters=custom_parameters,
    )

    if args.physics_engine != gymapi.SIM_PHYSX:
        raise RuntimeError("Sphere PPO test requires PhysX")

    torch.manual_seed(args.seed)

    SphereNavigationCfg.env.num_envs = args.num_envs
    cfg = SphereNavigationCfg()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.num_subscenes = args.subscenes
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.max_gpu_contact_pairs = 2 ** 20

    env = SphereNavigationEnv(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=True,
        episode_length=args.episode_length,
    )

    train_cfg = build_train_cfg()
    train_cfg["runner"]["max_iterations"] = args.iterations

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.expanduser("~/legged_workspace/logs/sphere_ppo_smoke"),
        timestamp,
    )
    os.makedirs(log_dir, exist_ok=True)

    print("===== SPHERE PPO CONFIG =====")
    print("NUM_ENVS={}".format(env.num_envs))
    print("NUM_OBS={}".format(env.num_obs))
    print("NUM_ACTIONS={}".format(env.num_actions))
    print("NUM_PRIVILEGED_OBS={}".format(env.num_privileged_obs))
    print("MAX_EPISODE_LENGTH={}".format(env.max_episode_length))
    print("ITERATIONS={}".format(args.iterations))
    print("DEVICE={}".format(env.device))
    print("LOG_DIR={}".format(log_dir))

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
    )

    checkpoint_path = os.path.join(
        log_dir,
        "model_{}.pt".format(args.iterations),
    )

    runner.save(checkpoint_path)

    checkpoint_ok = (
        os.path.isfile(checkpoint_path)
        and os.path.getsize(checkpoint_path) > 0
    )

    policy = runner.get_inference_policy(device=env.device)
    observations = env.get_observations()

    with torch.no_grad():
        actions = policy(observations)

    action_shape_ok = tuple(actions.shape) == (
        env.num_envs,
        env.num_actions,
    )
    action_finite = bool(torch.isfinite(actions).all().item())

    print("===== SPHERE PPO RESULT =====")
    print("CHECKPOINT_PATH={}".format(checkpoint_path))
    print("CHECKPOINT_OK={}".format(checkpoint_ok))
    print("ACTION_SHAPE={}".format(tuple(actions.shape)))
    print("ACTION_SHAPE_OK={}".format(action_shape_ok))
    print("ACTION_FINITE={}".format(action_finite))
    print(
        "TOTAL_ENV_SUCCESSES={}".format(
            int(env.success_counter.sum().item())
        )
    )
    print(
        "TOTAL_ENV_RESETS={}".format(
            int(env.reset_counter.sum().item())
        )
    )

    env.gym.destroy_sim(env.sim)

    if checkpoint_ok and action_shape_ok and action_finite:
        print("SPHERE_PPO_SMOKE=PASS")
        return 0

    print("SPHERE_PPO_SMOKE=FAIL")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            "SPHERE_PPO_SMOKE=FAIL_EXCEPTION:{}:{}".format(
                type(exc).__name__,
                exc,
            )
        )
        raise
