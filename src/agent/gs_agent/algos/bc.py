import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

import torch

from gs_agent.algos.config.schema import BCArgs
from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy
from gs_agent.buffers.bc_buffer import BCBuffer
from gs_agent.buffers.config.schema import BCBufferKey
from gs_agent.modules.models import NetworkFactory
from gs_agent.modules.policies import DeterministicPolicy, GaussianPolicy

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")
"""Default device for the algorithm."""

_DEQUE_MAXLEN: Final[int] = 100
"""Max length of the deque for storing episode statistics."""


class BC(BaseAlgo):
    """
    Behavior Cloning (BC) algorithm implementation.
    """

    def __init__(
        self, env: BaseEnvWrapper, cfg: BCArgs, device: torch.device = _DEFAULT_DEVICE
    ) -> None:
        super().__init__(env, cfg, device)
        self._actor_obs_dim = self.env.actor_obs_dim
        self._action_dim = self.env.action_dim
        #
        self._num_envs = self.env.num_envs
        self._num_steps = cfg.max_buffer_size
        #

        self.current_iter = 0
        self._rewbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._lenbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._curr_reward_sum = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.float
        )
        self._curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)

        # Load teacher config if provided (needed for teacher obs dim)
        if cfg.teacher_config_path is not None:
            self._load_teacher_config(cfg.teacher_config_path)
        else:
            self._teacher_obs_dim = self._actor_obs_dim  # Use student obs dim as fallback
            print(
                f"Teacher config not provided, using student observation dimension: {self._teacher_obs_dim}"
            )

        # Build actor network
        self._build_actor()
        if cfg.teacher_path.is_file():
            self._build_teacher(cfg.teacher_path)
        else:
            self._teacher = None
            print("Teacher network not provided.")
        self._build_rollouts()

    def _build_actor(self) -> None:
        policy_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.policy_backbone,
            input_dim=self._actor_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        print(f"Policy backbone: {policy_backbone}")
        self._actor = DeterministicPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
        ).to(self.device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self.cfg.lr)

    def _load_teacher_config(self, teacher_config_path: Path) -> None:
        """Load teacher environment config from yaml file."""
        if not teacher_config_path.is_file():
            raise ValueError(f"Teacher config path must be a file: {teacher_config_path}")

        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path as PathLib

        # Add examples to path to import utils
        examples_path = PathLib(__file__).parent.parent.parent.parent / "examples"
        if str(examples_path) not in sys.path:
            sys.path.insert(0, str(examples_path))

        # Import the appropriate config class based on environment type
        from gs_env.sim.envs.config.schema import MotionEnvArgs
        from utils import yaml_to_config  # type: ignore

        # Try to load as different config types
        self._teacher_env_args = yaml_to_config(teacher_config_path, MotionEnvArgs)

        # Compute teacher observation dimension
        teacher_obs, _ = self.env.get_observations(obs_args=self._teacher_env_args)
        self._teacher_obs_dim = teacher_obs.shape[-1]
        print(f"Teacher observation dimension: {self._teacher_obs_dim}")

    def _build_teacher(self, teacher_path: Path) -> None:
        teacher_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.teacher_backbone,
            input_dim=self._teacher_obs_dim,  # Use teacher obs dim instead of student
            output_dim=self._action_dim,
            device=self.device,
        )
        self._teacher = GaussianPolicy(
            policy_backbone=teacher_backbone,
            action_dim=self._action_dim,
        ).to(self.device)
        self._teacher.load_state_dict(
            torch.load(teacher_path, map_location=self.device)["actor_state_dict"]
        )
        self._teacher.eval()

    def _build_rollouts(self) -> None:
        self._rollouts = BCBuffer(
            num_envs=self._num_envs,
            max_steps=self._num_steps,
            obs_size=self._actor_obs_dim,  # TODO: add depth_shape and rgb_shape
            action_size=self._action_dim,
            device=self.device,
        )

    def _collect_rollouts(self, num_steps: int) -> dict[str, Any]:
        """Collect rollouts."""
        assert self._teacher is not None, "Teacher network not built"
        with torch.inference_mode():
            obs, _ = self.env.get_observations()  # Unpack actor and critic obs, only use actor
            termination_buffer = []
            reward_terms_buffer = []
            info_buffer = []
            # collect rollouts and compute returns & advantages
            for _step in range(num_steps):
                student_actions = self._actor(obs)
                # Get teacher observations if teacher config is provided
                if hasattr(self, "_teacher_env_args"):
                    teacher_obs, _ = self.env.get_observations(obs_args=self._teacher_env_args)
                else:
                    teacher_obs = obs  # Use student obs if no teacher config
                teacher_action, _ = self._teacher(teacher_obs, deterministic=True)
                # Step environment
                next_obs, reward, terminated, truncated, _extra_infos = self.env.step(
                    student_actions
                )

                # TODO: to use different observations for behavior cloning
                # not share the same observation space with the actor
                # all tensors are of shape: [num_envs, dim]
                transition = {
                    BCBufferKey.OBSERVATIONS: obs,
                    BCBufferKey.ACTIONS: teacher_action,
                }
                self._rollouts.append(transition)
                # Update episode tracking - handle reward and done sequences
                # Extract tensors from reward and done objects
                self._curr_reward_sum += reward.squeeze(-1)
                self._curr_ep_len += 1

                # Update termination buffer
                termination_buffer.append(_extra_infos["termination"])
                reward_terms_buffer.append(_extra_infos["reward_terms"])
                if "info" in _extra_infos:
                    info_buffer.append(_extra_infos["info"])
                # Check for episode completions and reset tracking
                done_mask = terminated.unsqueeze(-1) | truncated.unsqueeze(-1)
                new_ids = (done_mask > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    # Vectorized environment
                    self._rewbuffer.extend(
                        self._curr_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self._lenbuffer.extend(self._curr_ep_len[new_ids][:, 0].cpu().numpy().tolist())
                    # Reset tracking
                    self._curr_reward_sum[new_ids] = 0
                    self._curr_ep_len[new_ids] = 0

                obs = next_obs

        mean_reward = 0.0
        mean_ep_len = 0.0
        if len(self._rewbuffer) > 0:
            mean_reward = statistics.mean(self._rewbuffer)
            mean_ep_len = statistics.mean(self._lenbuffer)
        mean_termination = {}
        mean_reward_terms = {}
        mean_info = {}
        if len(termination_buffer) > 0:
            for key in termination_buffer[0].keys():
                terminations = torch.stack([termination[key] for termination in termination_buffer])
                mean_termination[key] = terminations.to(torch.float).mean().item()
        if len(reward_terms_buffer) > 0:
            for key in reward_terms_buffer[0].keys():
                reward_terms = torch.stack(
                    [reward_term[key] for reward_term in reward_terms_buffer]
                )
                mean_reward_terms[key] = reward_terms.mean().item()
        if len(info_buffer) > 0:
            for key in info_buffer[0].keys():
                infos = torch.tensor([info[key] for info in info_buffer])
                mean_info[key] = infos.mean().item()
        return {
            "mean_reward": mean_reward,
            "mean_ep_len": mean_ep_len,
            "termination": mean_termination,
            "reward_terms": mean_reward_terms,
            "info": mean_info,
        }

    def _train_one_batch(self, mini_batch: dict[BCBufferKey, torch.Tensor]) -> dict[str, Any]:
        """Train one batch of rollouts."""
        obs = mini_batch[BCBufferKey.OBSERVATIONS]
        act = mini_batch[BCBufferKey.ACTIONS]
        student_actions = self._actor(obs)
        loss = (act - student_actions).pow(2).mean()
        # Optimization step
        self._actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.max_grad_norm)
        self._actor_optimizer.step()
        # Log losses
        return {"loss": loss.item()}

    def train_one_iteration(self) -> dict[str, Any]:
        """Update policy using the collected experience."""
        t0 = time.time()
        rollout_infos = self._collect_rollouts(num_steps=self._num_steps)
        t1 = time.time()
        rollouts_time = t1 - t0
        fps = (self._num_steps * self._num_envs / rollouts_time) if rollouts_time > 0 else 0

        train_metrics_list: list[dict[str, Any]] = []
        for mini_batch in self._rollouts.minibatch_gen(
            batch_size=self.cfg.batch_size,
            num_epochs=self.cfg.num_epochs,
            max_num_batches=self.cfg.max_num_batches,
        ):
            metrics = self._train_one_batch(mini_batch)
            train_metrics_list.append(metrics)
        t2 = time.time()
        train_time = t2 - t1

        iteration_infos = {
            "rollout": {
                "mean_reward": rollout_infos["mean_reward"],
                "mean_length": rollout_infos["mean_ep_len"],
            },
            "train": {
                "imitation_loss": statistics.mean(
                    [metrics["loss"] for metrics in train_metrics_list]
                ),
            },
            "speed": {
                "rollout_time": rollouts_time,
                "rollout_fps": fps,
                "train_time": train_time,
                "rollout_step": self._num_steps * self._num_envs,
            },
            "termination": rollout_infos["termination"],
            "reward_terms": rollout_infos["reward_terms"],
            "info": rollout_infos["info"],
        }
        return iteration_infos

    def update_curriculum(
        self, current_iteration: int, total_iterations: int, **kwargs: Any
    ) -> None:
        pass

    def save(self, path: Path, infos: dict[str, Any] | None = None) -> None:
        saved_dict = {
            "actor_state_dict": self._actor.state_dict(),
            "optimizer_state_dict": self._actor_optimizer.state_dict(),
            "iter": self.current_iter,
        }
        if infos is not None:
            saved_dict.update(infos)
        torch.save(saved_dict, path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._actor.load_state_dict(checkpoint["actor_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["iter"]

    def train_mode(self) -> None:
        self._actor.train()

    def eval_mode(self) -> None:
        self._actor.eval()

    def get_inference_policy(self, device: torch.device | None = None) -> Policy:
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        return self._actor
