# Genesis Environment (`gs_env`)

A simulation environment framework built on top of Genesis, providing high-performance robotics simulation with support for manipulation tasks, various robot platforms, and flexible scene configurations.

## Overview

The `gs_env` module provides a unified interface for creating and managing simulation environments, supporting both Genesis-based physics simulation and real-world robot interfaces. It's designed for reinforcement learning applications with a focus on robotics and manipulation tasks.

## Key Features

- **High-Performance Simulation**: Built on Genesis physics engine with GPU acceleration
- **Modular Design**: Composable components for robots, scenes, objects, and sensors
- **Multi-Environment Support**: Batch processing for efficient RL training
- **Flexible Configuration**: Pydantic-based configuration system
- **Real-World Integration**: Support for both simulation and real robot interfaces
- **Rich Reward Systems**: Built-in reward functions for common robotics tasks

## Module Structure

### Core Components

#### `common/`
Shared utilities and base classes:
- **`bases/`**: Abstract base classes for environments, robots, objects, scenes, and sensors
- **`rewards/`**: Reward function implementations (ActionL2Penalty, KeypointsAlign)
- **`utils/`**: Mathematical utilities, asset management, and miscellaneous helpers

#### `sim/`
Simulation environment implementations:

##### `envs/`
Environment implementations:
- **`manipulation/`**: Manipulation task environments
  - **`goal_reaching_env.py`**: Goal-reaching task for manipulators
- **`config/`**: Environment configuration schemas and registry

##### `robots/`
Robot platform implementations:
- **`manipulators.py`**: Manipulator robot implementations (Franka)
- **`so101_robot.py`**: SO101 robot implementation
- **`config/`**: Robot configuration schemas and registry

##### `scenes/`
Scene configurations:
- **`flat_scene.py`**: Flat tabletop scene implementation
- **`config/`**: Scene configuration schemas and registry

##### `objects/`
Simulated objects:
- **`config/`**: Object configuration schemas and registry

##### `sensors/`
Sensor implementations:
- **`camera.py`**: Camera sensor implementation
- **`config/`**: Sensor configuration schemas and registry

#### `interface/`
Real-world robot interfaces:
- **`teleop_wrapper.py`**: Teleoperation interface wrapper

#### `real/`
Real robot implementations (placeholder for future development)

## Quick Start

### Creating a Goal-Reaching Environment

```python
import torch
from gs_env.sim.envs.manipulation.goal_reaching_env import GoalReachingEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry

# Create environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = GoalReachingEnv(
    args=EnvArgsRegistry["goal_reach_default"],
    num_envs=2048,
    show_viewer=False,
    device=device
)

# Environment interaction
obs = env.get_observations()
reward, info = env.get_reward()
env.apply_action(action)
```

### Custom Environment Configuration

```python
from gs_env.sim.envs.config.schema import EnvArgs, RobotArgs, SceneArgs

# Create custom configuration
env_args = EnvArgs(
    robot_args=RobotArgs(
        # Robot-specific parameters
    ),
    scene_args=SceneArgs(
        # Scene-specific parameters
    ),
    reward_args={
        "rew_actions": 0.01,
        "rew_keypoints": 1.0
    }
)
```

## Environment Types

### Goal Reaching Environment

A manipulation task where a robot must reach a target position and orientation:

**Observation Space:**
- `pose_vec`: 3D position difference (3D)
- `ee_quat`: End-effector quaternion (4D)
- `ref_position`: Target position (3D)
- `ref_quat`: Target quaternion (4D)

**Action Space:**
- Joint position commands (6D for Franka robot)

**Reward Function:**
- Action L2 penalty
- Keypoint alignment reward

## Robot Platforms

### Franka Robot
- 7-DOF manipulator
- Gripper control
- Joint position and velocity control
- End-effector pose control

### SO101 Robot
- Specialized robot implementation
- Custom configuration options

## Scene Types

### Flat Scene
- Tabletop environment
- Configurable dimensions
- Support for multiple environments
- Optional viewer integration

## Reward Functions

### ActionL2Penalty
Penalizes large actions to encourage smooth motion:
```python
reward = -scale * ||action||²
```

### KeypointsAlign
Rewards alignment between current and target poses:
```python
reward = scale * alignment_score
```

## Configuration System

All components use Pydantic-based configuration with validation:

```python
from gs_env.sim.robots.config.schema import RobotArgs

robot_config = RobotArgs(
    # Configuration parameters with automatic validation
)
```

Pre-configured settings are available in registries:
```python
from gs_env.sim.envs.config.registry import EnvArgsRegistry

env_args = EnvArgsRegistry["goal_reach_default"]
```

## Multi-Environment Support

The framework supports batch processing for efficient RL training:

```python
# Create multiple parallel environments
env = GoalReachingEnv(
    args=env_args,
    num_envs=2048,  # 2048 parallel environments
    device=device
)

# All operations work on batches
obs = env.get_observations()  # Shape: (2048, obs_dim)
rewards, info = env.get_reward()  # Shape: (2048,)
```

## Device Management

Automatic device detection and tensor management:

```python
# Automatic device detection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# All tensors are automatically placed on the correct device
```

## Integration with RL Agents

The environment integrates seamlessly with the `gs_agent` framework:

```python
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper

# Wrap environment for RL training
wrapped_env = GenesisEnvWrapper(env, device=device)

# Use with RL algorithms
from gs_agent.algos.ppo import PPO
ppo = PPO(env=wrapped_env, cfg=config, device=device)
```

## Examples

See the `examples/` directory for complete usage examples:
- `run_ppo_gs.py`: Training with Genesis environments

## Dependencies

- Genesis (physics simulation)
- PyTorch
- Gymnasium
- Pydantic
- NumPy

## Performance Considerations

- Use GPU acceleration when available (CUDA/MPS)
- Batch size affects memory usage and training speed
- Viewer mode reduces performance for training
- Consider using smaller batch sizes for debugging

## Contributing

When adding new environments or components:
1. Inherit from appropriate base classes
2. Follow the configuration schema pattern
3. Add comprehensive type hints
4. Include reward function implementations
5. Update this README with usage examples
