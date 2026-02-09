import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from gs_agent.algos.config.schema import LearningRateType, PPOArgs
from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy
from gs_agent.buffers.config.schema import GAEBufferKey
from gs_agent.buffers.gae_buffer import GAEBuffer
from gs_agent.modules.critics import StateValueFunction
from gs_agent.modules.models import NetworkFactory
from gs_agent.modules.policies import GaussianPolicy

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")
"""Default device for the algorithm."""

_DEQUE_MAXLEN: Final[int] = 100
"""Max length of the deque for storing episode statistics."""


class PPO(BaseAlgo):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """

    def __init__(
        self, env: BaseEnvWrapper, cfg: PPOArgs, device: torch.device = _DEFAULT_DEVICE
    ) -> None:
        super().__init__(env, cfg, device)
        self._actor_obs_dim = self.env.actor_obs_dim
        self._critic_obs_dim = self.env.critic_obs_dim
        self._action_dim = self.env.action_dim

        self._num_envs = self.env.num_envs
        self._num_steps = cfg.rollout_length

        self.current_iter = 0
        self._rewbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._lenbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._curr_reward_sum = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.float
        )
        self._curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)

        self._current_lr = cfg.lr
        self._current_entropy_coef = cfg.entropy_coef

        self.use_clipped_value_loss = cfg.use_clipped_value_loss

        # Freeze actor and critic
        self._freeze_actor = False
        self._freeze_critic = False

        self._build_actor_critic()
        self._build_rollouts()

    def _build_actor_critic(self) -> None:
        policy_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.policy_backbone,
            input_dim=self._actor_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        print(f"Policy backbone: {policy_backbone}")

        self._actor = GaussianPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
        ).to(self.device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._current_lr)

        critic_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.critic_backbone,
            input_dim=self._critic_obs_dim,
            output_dim=1,
            device=self.device,
        )
        print(f"Critic backbone: {critic_backbone}")
        self._critic = StateValueFunction(critic_backbone).to(self.device)
        value_lr = self.cfg.value_lr if self.cfg.value_lr is not None else self.cfg.lr
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=value_lr)

        print("Device: ", self.device)

    def _build_rollouts(self) -> None:
        self._rollouts = GAEBuffer(
            num_envs=self._num_envs,
            max_steps=self._num_steps,
            actor_obs_size=self._actor_obs_dim,
            critic_obs_size=self._critic_obs_dim,
            action_size=self._action_dim,
            device=self.device,
        )

    def _update_learning_rate(self, kl_mean: float) -> None:
        """Update learning rate adaptively based on KL divergence."""
        if self.cfg.lr_type == LearningRateType.ADAPTIVE:
            # If KL is too high, reduce learning rate
            if kl_mean > self.cfg.target_kl * 1.5:  # 1.5x threshold
                self._current_lr = max(
                    self._current_lr / self.cfg.lr_adaptive_factor, self.cfg.lr_min
                )
            # If KL is too low, increase learning rate
            elif kl_mean < self.cfg.target_kl * 0.5:  # 0.5x threshold
                self._current_lr = min(
                    self._current_lr * self.cfg.lr_adaptive_factor, self.cfg.lr_max
                )

            # Update only actor optimizer learning rate
            for param_group in self._actor_optimizer.param_groups:
                param_group["lr"] = self._current_lr

    def _collect_rollouts(self, num_steps: int) -> dict[str, Any]:
        """Collect rollouts from the environment."""
        with torch.inference_mode():
            actor_obs, critic_obs = self.env.get_observations()
            termination_buffer = []
            reward_terms_buffer = []
            info_buffer = []
            # collect rollouts and compute returns & advantages
            for _step in range(num_steps):
                action, log_prob, mu, sigma = self._actor.forward_with_dist_params(actor_obs)
                # Step environment
                _, reward, terminated, truncated, _extra_infos = self.env.step(action)
                next_actor_obs, next_critic_obs = self.env.get_observations()

                # add next value to reward of truncated steps
                if "time_outs" in _extra_infos:
                    time_outs = _extra_infos["time_outs"]
                    next_values = self._critic(_extra_infos["observations"]["critic"])
                    reward = reward + next_values * time_outs

                # all tensors are of shape: [num_envs, dim]
                transition = {
                    GAEBufferKey.ACTOR_OBS: actor_obs,
                    GAEBufferKey.CRITIC_OBS: critic_obs,
                    GAEBufferKey.ACTIONS: action,
                    GAEBufferKey.REWARDS: reward,
                    GAEBufferKey.DONES: terminated,
                    GAEBufferKey.VALUES: self._critic(critic_obs),
                    GAEBufferKey.ACTION_LOGPROBS: log_prob,
                    GAEBufferKey.MU: mu,
                    GAEBufferKey.SIGMA: sigma,
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

                actor_obs, critic_obs = next_actor_obs, next_critic_obs
        with torch.no_grad():
            last_value = self._critic(critic_obs)
            self._rollouts.set_final_value(last_value)

        mean_reward = 0.0
        mean_ep_len = 0.0
        if len(self._rewbuffer) > 0:
            mean_reward = statistics.mean(self._rewbuffer)
            mean_ep_len = statistics.mean(self._lenbuffer)
        # import ipdb; ipdb.set_trace()
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

    def _train_one_batch(self, mini_batch: dict[GAEBufferKey, torch.Tensor]) -> dict[str, Any]:
        """Train one batch of rollouts."""
        actor_obs = mini_batch[GAEBufferKey.ACTOR_OBS]
        critic_obs = mini_batch[GAEBufferKey.CRITIC_OBS]
        act = mini_batch[GAEBufferKey.ACTIONS]
        old_log_prob = mini_batch[GAEBufferKey.ACTION_LOGPROBS]
        old_mu = mini_batch[GAEBufferKey.MU]
        old_sigma = mini_batch[GAEBufferKey.SIGMA]
        advantage = mini_batch[GAEBufferKey.ADVANTAGES]
        returns = mini_batch[GAEBufferKey.RETURNS]
        target_values = mini_batch[GAEBufferKey.VALUES]

        #
        new_log_prob = self._actor.evaluate_log_prob(actor_obs, act)
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = -advantage * ratio
        surr2 = -advantage * torch.clamp(
            ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
        )
        policy_loss = torch.max(surr1, surr2).mean()

        # Rename approx_kl to log_ratio
        log_ratio = (new_log_prob - old_log_prob).mean()

        # Calculate true KL divergence between old and new distributions
        # Get new distribution parameters
        dist = self._actor.dist_from_obs(actor_obs)
        new_mu = dist.mean
        new_sigma = dist.stddev

        # KL divergence between two Gaussian distributions
        # KL(P||Q) = 0.5 * [log(det(Σ_Q)/det(Σ_P)) - d + tr(Σ_Q^{-1} * Σ_P) + (μ_Q - μ_P)^T * Σ_Q^{-1} * (μ_Q - μ_P)]
        # For diagonal Gaussians: KL = 0.5 * [sum(log(σ_Q^2/σ_P^2)) - d + sum(σ_P^2/σ_Q^2) + sum((μ_Q - μ_P)^2/σ_Q^2)]
        kl_div = 0.5 * (
            torch.log(new_sigma**2 / (old_sigma**2 + 1e-8)).sum(
                dim=-1, keepdim=True
            )  # log ratio of variances
            - new_mu.shape[-1]  # dimension
            + (old_sigma**2 / (new_sigma**2 + 1e-8)).sum(dim=-1, keepdim=True)  # trace term
            + ((new_mu - old_mu) ** 2 / (new_sigma**2 + 1e-8)).sum(
                dim=-1, keepdim=True
            )  # mean difference term
        )
        kl_mean = kl_div.mean()

        # Calculate value loss
        values = self._critic(critic_obs)
        if self.use_clipped_value_loss:
            clipped_values = target_values + (values - target_values).clamp(
                -self.cfg.clip_ratio, self.cfg.clip_ratio
            )
            value_loss = (values - returns).pow(2)
            clipped_value_loss = (clipped_values - returns).pow(2)
            value_loss = torch.max(value_loss, clipped_value_loss).mean()
        else:
            value_loss = (returns - values).pow(2).mean()

        # Calculate entropy loss
        entropy = self._actor.entropy_on(actor_obs)
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.cfg.value_loss_coef * value_loss
            - self._current_entropy_coef * entropy_loss
        )

        # Optimization step
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
        if not self._freeze_critic:
            self._critic_optimizer.step()
        if not self._freeze_actor:
            self._actor_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "log_ratio": log_ratio.item(),
            "kl_mean": kl_mean.item(),
            "mean_std": self._actor.action_std.mean().item(),
            "max_std": self._actor.action_std.max().item(),
        }

    def train_one_iteration(self) -> dict[str, Any]:
        """Train one iteration."""
        # collect rollouts
        t0 = time.time()
        rollout_infos = self._collect_rollouts(num_steps=self._num_steps)
        t1 = time.time()
        rollouts_time = t1 - t0
        fps = (self._num_steps * self._num_envs / rollouts_time) if rollouts_time > 0 else 0

        train_metrics_list: list[dict[str, Any]] = []
        for mini_batch in self._rollouts.minibatch_gen(
            num_mini_batches=self.cfg.num_mini_batches,
            num_epochs=self.cfg.num_epochs,
        ):
            metrics = self._train_one_batch(mini_batch)
            train_metrics_list.append(metrics)
        t2 = time.time()
        train_time = t2 - t1

        # Update learning rate adaptively based on KL divergence
        avg_kl_mean = statistics.mean([metrics["kl_mean"] for metrics in train_metrics_list])
        if not self._freeze_actor:
            self._update_learning_rate(avg_kl_mean)

        self._rollouts.reset()

        iteration_infos = {
            "rollout": {
                "mean_reward": rollout_infos["mean_reward"],
                "mean_length": rollout_infos["mean_ep_len"],
            },
            "train": {
                "policy_loss": statistics.mean(
                    [metrics["policy_loss"] for metrics in train_metrics_list]
                ),
                "value_loss": statistics.mean(
                    [metrics["value_loss"] for metrics in train_metrics_list]
                ),
                "entropy_loss": statistics.mean(
                    [metrics["entropy_loss"] for metrics in train_metrics_list]
                ),
                "log_ratio": statistics.mean(
                    [metrics["log_ratio"] for metrics in train_metrics_list]
                ),
                "kl_mean": statistics.mean([metrics["kl_mean"] for metrics in train_metrics_list]),
                "mean_std": statistics.mean(
                    [metrics["mean_std"] for metrics in train_metrics_list]
                ),
                "max_std": statistics.mean([metrics["max_std"] for metrics in train_metrics_list]),
                "learning_rate": self._current_lr,
                "entropy_coef": self._current_entropy_coef,
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
        self._current_entropy_coef = self.cfg.entropy_coef * (
            1 - current_iteration / total_iterations
        )

    def save(self, path: Path) -> None:
        """Save the algorithm to a file."""
        saved_dict = {
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optimizer_state_dict": self._actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_optimizer.state_dict(),
            "iter": self.current_iter,
        }
        torch.save(saved_dict, path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        """Load the algorithm from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self._actor.load_state_dict(checkpoint["actor_state_dict"])
        self._critic.load_state_dict(checkpoint["critic_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.current_iter = checkpoint["iter"]

    def train_mode(self) -> None:
        """Set the algorithm to train mode."""
        self._actor.train()

    def eval_mode(self) -> None:
        """Set the algorithm to eval mode."""
        self._actor.eval()

    def get_inference_policy(self, device: torch.device | None = None) -> Policy:
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        policy = self._actor
        return policy

    def freeze_actor(self) -> None:
        """Freeze the actor."""
        self._freeze_actor = True

    def freeze_critic(self) -> None:
        """Freeze the critic."""
        self._freeze_critic = True

    def unfreeze_actor(self) -> None:
        """Unfreeze the actor."""
        self._freeze_actor = False

    def unfreeze_critic(self) -> None:
        """Unfreeze the critic."""
        self._freeze_critic = False
