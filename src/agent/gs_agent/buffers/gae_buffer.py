from collections.abc import Iterator
from typing import Final

import torch
from tensordict import TensorDict

from gs_agent.bases.buffer import BaseBuffer
from gs_agent.buffers.config.schema import GAEBufferKey

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    final_value: torch.Tensor | None = None,
    truncation_mask: torch.Tensor | None = None,  # optional, shape [T, B, 1]
    normalize_advantages: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) with optional support for truncated episodes.

    Args:
        rewards: Reward object containing reward tensor [time_dim, batch_dim, 1]
        values: StateValue object containing value tensor [time_dim, batch_dim, 1]
        dones: Done object containing done flags [time_dim, batch_dim, 1]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        final_value: Final value estimate for bootstrapping [batch_dim, 1]. If None, uses zero.
        truncation_mask: Optional mask tensor [time_dim, batch_dim, 1], where 1 indicates valid step and 0 indicates truncation.
        normalize_advantages: Whether to normalize the advantages

    Returns:
        Tuple of (Advantage, Returns), both of shape [time_dim, batch_dim, 1]
    """
    masks = 1.0 - dones.float()  # Standard mask for episode termination

    if truncation_mask is not None:
        masks = (masks.bool() & truncation_mask.bool()).float()

    time_dim, batch_dim, _ = rewards.shape

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = torch.zeros(batch_dim, 1, device=rewards.device, dtype=rewards.dtype)

    # Append final value for bootstrapping
    next_values = torch.cat(
        [
            values[1:],
            (
                final_value.unsqueeze(0)
                if final_value is not None
                else torch.zeros(1, batch_dim, 1, device=rewards.device)
            ),
        ],
        dim=0,
    )

    # Compute deltas
    deltas = rewards + gamma * next_values * masks - values

    # Compute GAE in reverse
    for t in reversed(range(time_dim)):
        gae = deltas[t] + gamma * gae_lambda * masks[t] * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


class GAEBuffer(BaseBuffer):
    """
    A fixed-size buffer for storing rollouts for multi-env on-policy training,
    with GAE/Generalized Advantage Estimation.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        actor_obs_size: int,
        critic_obs_size: int,
        action_size: int,
        device: torch.device = _DEFAULT_DEVICE,
        gae_gamma: float = 0.98,
        gae_lam: float = 0.95,
    ) -> None:
        """
        Args:
            num_envs (int): Number of parallel environments.
            max_steps (int): Maximum number of steps to store in the buffer.
            actor_obs_size (int): Dimension of the actor's observation space.
            action_size (int): Dimension of the action space.
            device (torch.device): Device to store the buffer on (default: CPU).
            gae_gamma (float): Discount factor for GAE.
            gae_lam (float): Lambda parameter for GAE.
        """
        super().__init__()
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._actor_obs_dim = actor_obs_size
        self._critic_obs_dim = critic_obs_size
        self._action_dim = action_size
        self._device = device

        # GAE parameters (configurable)
        self._gae_gamma = gae_gamma
        self._gae_lam = gae_lam

        self._normalize_adv = True  # Whether to normalize advantages

        # Track current size dynamically
        self._idx, self._idx_hidden = 0, 0

        self._final_value = None

        # Initialize buffer
        self._buffer = self._init_buffers(actor_obs_size, critic_obs_size, action_size)

    def _init_buffers(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int) -> TensorDict:
        max_steps, num_envs = self._max_steps, self._num_envs
        buffer = TensorDict(
            {
                GAEBufferKey.ACTOR_OBS: torch.zeros(max_steps, num_envs, actor_obs_dim),
                GAEBufferKey.CRITIC_OBS: torch.zeros(max_steps, num_envs, critic_obs_dim),
                GAEBufferKey.ACTIONS: torch.zeros(max_steps, num_envs, action_dim),
                GAEBufferKey.REWARDS: torch.zeros(max_steps, num_envs, 1),
                GAEBufferKey.DONES: torch.zeros(max_steps, num_envs, 1).byte(),
                GAEBufferKey.VALUES: torch.zeros(max_steps, num_envs, 1),
                GAEBufferKey.ACTION_LOGPROBS: torch.zeros(max_steps, num_envs, 1),
                GAEBufferKey.MU: torch.zeros(max_steps, num_envs, action_dim),
                GAEBufferKey.SIGMA: torch.zeros(max_steps, num_envs, action_dim),
            },
            batch_size=[self._max_steps, self._num_envs],
            device=self._device,
        )
        return buffer

    def reset(self) -> None:
        self._idx = 0
        self._final_value = None

    def append(self, transition: dict[GAEBufferKey, torch.Tensor]) -> None:
        if self._idx >= self._max_steps:
            raise ValueError(f"Buffer full! Cannot append more than {self._max_steps} steps.")
        idx = self._idx
        self._buffer[GAEBufferKey.ACTOR_OBS][idx] = transition[GAEBufferKey.ACTOR_OBS]
        self._buffer[GAEBufferKey.CRITIC_OBS][idx] = transition[GAEBufferKey.CRITIC_OBS]
        self._buffer[GAEBufferKey.ACTIONS][idx] = transition[GAEBufferKey.ACTIONS]
        self._buffer[GAEBufferKey.REWARDS][idx] = transition[GAEBufferKey.REWARDS]
        self._buffer[GAEBufferKey.DONES][idx] = transition[GAEBufferKey.DONES]
        self._buffer[GAEBufferKey.VALUES][idx] = transition[GAEBufferKey.VALUES]
        self._buffer[GAEBufferKey.ACTION_LOGPROBS][idx] = transition[GAEBufferKey.ACTION_LOGPROBS]
        self._buffer[GAEBufferKey.MU][idx] = transition[GAEBufferKey.MU]
        self._buffer[GAEBufferKey.SIGMA][idx] = transition[GAEBufferKey.SIGMA]

        # Increment index
        self._idx += 1

    def set_final_value(self, final_value: torch.Tensor) -> None:
        """Set the final value for bootstrapping incomplete episodes.

        This is used to bootstrap the value function for episodes that didn't
        naturally terminate, which is important for short horizon policy updates.

        Args:
            final_value: Value estimate for the final observation [B, 1]
        """
        self._final_value = final_value

    def is_full(self) -> bool:
        return self._idx >= self._max_steps

    def __len__(self) -> int:
        return self._idx * self._num_envs

    def minibatch_gen(
        self, num_mini_batches: int, num_epochs: int, shuffle: bool = True
    ) -> Iterator[dict[GAEBufferKey, torch.Tensor]]:
        advantages, returns = compute_gae(
            rewards=self._buffer[GAEBufferKey.REWARDS],
            values=self._buffer[GAEBufferKey.VALUES],
            dones=self._buffer[GAEBufferKey.DONES],
            final_value=self._final_value,
            gamma=self._gae_gamma,
            gae_lambda=self._gae_lam,
        )

        # Get total number of timesteps and batch size
        total_samples = self._max_steps * self._num_envs
        # Create indices for all samples on the same device as the tensors
        base_indices = torch.arange(total_samples, device=self._device)

        for _epoch in range(num_epochs):
            # Split into minibatches
            if shuffle:
                perm_indices = base_indices[torch.randperm(total_samples, device=self._device)]
            else:
                perm_indices = base_indices

            buckets = torch.tensor_split(perm_indices, num_mini_batches)  # not torch.split here
            for bucket in buckets:
                if bucket.numel() == 0:
                    continue

                t_idx = bucket // self._num_envs
                b_idx = bucket % self._num_envs
                mini_batch_size = bucket.numel()
                yield {
                    GAEBufferKey.ACTOR_OBS: self._buffer[GAEBufferKey.ACTOR_OBS][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    GAEBufferKey.CRITIC_OBS: self._buffer[GAEBufferKey.CRITIC_OBS][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    GAEBufferKey.ACTIONS: self._buffer[GAEBufferKey.ACTIONS][t_idx, b_idx].reshape(
                        mini_batch_size, -1
                    ),
                    GAEBufferKey.REWARDS: self._buffer[GAEBufferKey.REWARDS][t_idx, b_idx].reshape(
                        mini_batch_size, -1
                    ),
                    GAEBufferKey.DONES: self._buffer[GAEBufferKey.DONES][t_idx, b_idx].reshape(
                        mini_batch_size, -1
                    ),
                    GAEBufferKey.VALUES: self._buffer[GAEBufferKey.VALUES][t_idx, b_idx].reshape(
                        mini_batch_size, -1
                    ),
                    GAEBufferKey.ACTION_LOGPROBS: self._buffer[GAEBufferKey.ACTION_LOGPROBS][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    GAEBufferKey.MU: self._buffer[GAEBufferKey.MU][t_idx, b_idx].reshape(
                        mini_batch_size, -1
                    ),
                    GAEBufferKey.SIGMA: self._buffer[GAEBufferKey.SIGMA][t_idx, b_idx].reshape(
                        mini_batch_size, -1
                    ),
                    GAEBufferKey.ADVANTAGES: advantages[t_idx, b_idx].reshape(mini_batch_size, -1),
                    GAEBufferKey.RETURNS: returns[t_idx, b_idx].reshape(mini_batch_size, -1),
                }
