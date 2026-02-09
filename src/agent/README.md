# Genesis Agent (`gs_agent`)

A robot learning agent framework built on top of PyTorch, designed to work with Genesis simulation environments and standard Gymnasium environments.

## Overview

The `gs_agent` module provides a complete RL training pipeline including algorithms, neural network architectures, experience buffers, and training runners. It's specifically designed to work with both Genesis simulation environments and standard Gymnasium environments.

## Key Features

- **RL Algorithms**: Currently supports PPO (Proximal Policy Optimization)
- **Flexible Architecture**: Modular design with pluggable components
- **Multi-Environment Support**: Works with both Genesis and Gymnasium environments
- **GPU Acceleration**: Automatic device detection (CUDA, MPS, CPU)
- **Comprehensive Logging**: Built-in support for Weights & Biases, CSV, and console logging
- **Configuration Management**: Pydantic-based configuration system with validation

## Module Structure

### Core Components

#### `algos/`
Contains reinforcement learning algorithms:
- **`ppo.py`**: Proximal Policy Optimization implementation with GAE (Generalized Advantage Estimation)

#### `bases/`
Abstract base classes defining the framework interfaces:
- **`algo.py`**: Base algorithm interface
- **`buffer.py`**: Experience buffer interface
- **`critic.py`**: Value function interface
- **`env_wrapper.py`**: Environment wrapper interface
- **`network_backbone.py`**: Neural network backbone interface
- **`policy.py`**: Policy network interface
- **`runner.py`**: Training runner interface

#### `buffers/`
Experience replay and advantage estimation:
- **`gae_buffer.py`**: GAE buffer for on-policy algorithms

#### `configs/`
Configuration management:
- **`registry.py`**: Pre-configured algorithm and runner settings
- **`schema.py`**: Pydantic schemas for configuration validation

#### `modules/`
Neural network implementations:
- **`critics.py`**: Value function networks
- **`models.py`**: Network factory and architecture definitions
- **`policies.py`**: Policy network implementations (Gaussian policies)

#### `runners/`
Training orchestration:
- **`onpolicy_runner.py`**: On-policy training runner with episode management

#### `utils/`
Utility functions:
- **`logger.py`**: Comprehensive logging system with W&B integration
- **`misc.py`**: Miscellaneous utilities
- **`policy_loader.py`**: Model loading and saving utilities
- **`torch_utils.py`**: PyTorch-specific utilities and device management

#### `wrappers/`
Environment adapters:
- **`gs_env_wrapper.py`**: Genesis environment wrapper
- **`gym_env_wrapper.py`**: Gymnasium environment wrapper

## Quick Start

### Training with Genesis Environment

```python
import torch
from gs_agent.algos.ppo import PPO
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.configs import PPO_GOAL_REACHING_MLP, RUNNER_GOAL_REACHING_MLP
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.manipulation.goal_reaching_env import GoalReachingEnv

# Create environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = GoalReachingEnv(args=env_args, num_envs=2048, device=device)
wrapped_env = GenesisEnvWrapper(env, device=device)

# Create PPO algorithm
ppo = PPO(env=wrapped_env, cfg=PPO_GOAL_REACHING_MLP, device=device)

# Create runner and train
runner = OnPolicyRunner(algorithm=ppo, runner_args=RUNNER_GOAL_REACHING_MLP, device=device)
runner.train()
```

### Training with Gymnasium Environment

```python
import gymnasium as gym
from gs_agent.wrappers.gym_env_wrapper import GymEnvWrapper

# Create Gymnasium environment
gym_env = gym.make("Pendulum-v1")
wrapped_env = GymEnvWrapper(gym_env, device=device)

# Use with PPO as above
```

## Configuration

The framework uses Pydantic for configuration management with built-in validation:

```python
from gs_agent.configs.schema import PPOArgs

# Create custom configuration
config = PPOArgs(
    learning_rate=3e-4,
    num_epochs=10,
    batch_size=64,
    # ... other parameters
)
```

Pre-configured settings are available in the registry:

```python
from gs_agent.configs import PPO_GOAL_REACHING_MLP, RUNNER_GOAL_REACHING_MLP
```

## Logging

The framework provides comprehensive logging capabilities:

```python
from gs_agent.utils.logger import configure

# Configure logging with multiple outputs
logger = configure(
    folder="./logs",
    format_strings=["stdout", "csv", "wandb"]
)
```

Supported logging formats:
- **stdout**: Console output
- **csv**: CSV file logging
- **wandb**: Weights & Biases integration

## Device Management

Automatic device detection with fallback:

```python
from gs_agent.utils.torch_utils import get_torch_device

device = get_torch_device()  # Automatically detects CUDA, MPS, or CPU
```

## Examples

See the `examples/` directory for complete training scripts:
- `run_ppo_gs.py`: Training with Genesis environments
- `run_ppo_gym.py`: Training with Gymnasium environments

## Dependencies

- PyTorch
- Pydantic
- Gymnasium
- Weights & Biases (optional)
- Genesis (for Genesis environments)

## Contributing

When adding new algorithms or components:
1. Inherit from the appropriate base class
2. Follow the existing naming conventions
3. Add comprehensive type hints
4. Include configuration schemas
5. Update this README with usage examples
