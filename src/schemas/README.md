# Genesis Schemas (`gs_schemas`)

Shared dataclasses, type definitions, and interfaces used across the Genesis Playground framework. This module provides a common foundation for type safety, validation, and configuration management throughout the entire codebase.

## Overview

The `gs_schemas` module serves as the central repository for all shared data structures, type definitions, and configuration schemas used across the Genesis Playground framework. It ensures consistency, type safety, and validation across all modules.

## Key Features

- **Type Safety**: Comprehensive type definitions and annotations
- **Validation**: Pydantic-based validation for all data structures
- **Consistency**: Shared interfaces and base types across modules
- **Documentation**: Self-documenting schemas with clear field descriptions
- **Extensibility**: Easy to extend with new types and interfaces

## Module Structure

### Core Components

#### `base_types.py`
Fundamental type definitions and utilities:
- **Pydantic Configuration**: Default configuration for all Genesis Pydantic models
- **Base Types**: Common type aliases and enums
- **Validation Utilities**: Helper functions for data validation
- **Serialization**: JSON serialization and deserialization utilities

## Quick Start

### Using Pydantic Configuration

```python
from pydantic.dataclasses import dataclass
from gs_schemas.base_types import genesis_pydantic_config

@dataclass(config=genesis_pydantic_config())
class MyDataClass:
    name: str
    value: float
    optional_field: str | None = None

# Automatic validation and type checking
data = MyDataClass(name="test", value=42.0)
```

### Custom Configuration

```python
from gs_schemas.base_types import genesis_pydantic_config

# Create custom configuration with specific settings
custom_config = genesis_pydantic_config(
    frozen=True,  # Make instances immutable
    arbitrary_types_allowed=True  # Allow arbitrary types
)

@dataclass(config=custom_config)
class ImmutableDataClass:
    # Fields will be validated and frozen
    pass
```

## Configuration Features

### Default Pydantic Configuration

The `genesis_pydantic_config()` function provides sensible defaults for all Genesis Pydantic models:

- **`extra="forbid"`**: Prevents extra fields not defined in the model
- **`validate_default=True`**: Validates default values against field types
- **`use_enum_values=True`**: Uses enum values instead of enum objects
- **`validate_assignment=True`**: Validates assignments to model fields

### Type Validation

All schemas include comprehensive type validation:

```python
from gs_schemas.base_types import genesis_pydantic_config
from pydantic.dataclasses import dataclass
from typing import Literal

@dataclass(config=genesis_pydantic_config())
class DeviceConfig:
    device_type: Literal["cuda", "mps", "cpu"]
    device_id: int = 0

    def __post_init__(self):
        if self.device_type == "cpu" and self.device_id != 0:
            raise ValueError("CPU device ID must be 0")

# Automatic validation
config = DeviceConfig(device_type="cuda", device_id=0)  # Valid
# config = DeviceConfig(device_type="cuda", device_id=-1)  # Raises validation error
```

## Integration with Other Modules

### Agent Module Integration

```python
from gs_schemas.base_types import genesis_pydantic_config
from gs_agent.configs.schema import PPOArgs

# PPOArgs uses Genesis Pydantic configuration
ppo_config = PPOArgs(
    learning_rate=3e-4,
    num_epochs=10,
    # ... other parameters with automatic validation
)
```

### Environment Module Integration

```python
from gs_env.sim.envs.config.schema import EnvArgs

# Environment configurations use Genesis schemas
env_config = EnvArgs(
    robot_args=robot_config,
    scene_args=scene_config,
    # ... validated automatically
)
```

## Best Practices

### Schema Design

1. **Use Descriptive Names**: Choose clear, descriptive names for all fields
2. **Add Type Hints**: Always include comprehensive type annotations
3. **Provide Defaults**: Include sensible default values where appropriate
4. **Validate Constraints**: Add validation logic for business rules
5. **Document Fields**: Use docstrings to document field purposes

### Example Schema

```python
from gs_schemas.base_types import genesis_pydantic_config
from pydantic.dataclasses import dataclass
from typing import Optional, Literal

@dataclass(config=genesis_pydantic_config())
class TrainingConfig:
    """Configuration for RL training sessions."""

    # Required fields
    algorithm: Literal["ppo", "sac", "td3"]
    num_epochs: int
    learning_rate: float

    # Optional fields with defaults
    batch_size: int = 64
    device: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration constraints."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
```

### Error Handling

```python
from pydantic import ValidationError

try:
    config = TrainingConfig(
        algorithm="invalid",  # Will raise ValidationError
        num_epochs=10,
        learning_rate=0.001
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Serialization

### JSON Serialization

```python
import json
from gs_schemas.base_types import genesis_pydantic_config
from pydantic.dataclasses import dataclass

@dataclass(config=genesis_pydantic_config())
class SerializableConfig:
    name: str
    value: float

config = SerializableConfig(name="test", value=42.0)

# Serialize to JSON
json_str = json.dumps(config.__dict__)

# Deserialize from JSON
loaded_data = json.loads(json_str)
loaded_config = SerializableConfig(**loaded_data)
```

## Dependencies

- **Pydantic**: Data validation and settings management
- **Python 3.8+**: Type hint support
- **typing**: Type annotations and generics

## Contributing

When adding new schemas:

1. **Follow Naming Conventions**: Use descriptive, consistent names
2. **Add Type Hints**: Include comprehensive type annotations
3. **Include Validation**: Add appropriate validation logic
4. **Write Documentation**: Document all fields and their purposes
5. **Add Tests**: Include unit tests for validation logic
6. **Update This README**: Add usage examples for new schemas

### Schema Template

```python
from gs_schemas.base_types import genesis_pydantic_config
from pydantic.dataclasses import dataclass
from typing import Optional

@dataclass(config=genesis_pydantic_config())
class NewSchema:
    """Description of the schema purpose."""

    # Required fields
    required_field: str

    # Optional fields with defaults
    optional_field: Optional[int] = None

    def __post_init__(self):
        """Add validation logic here."""
        pass
```

## Migration Guide

When updating existing schemas:

1. **Maintain Backward Compatibility**: Use Optional fields for new additions
2. **Deprecate Gradually**: Mark old fields as deprecated before removal
3. **Update Validation**: Ensure new validation doesn't break existing usage
4. **Test Thoroughly**: Verify all dependent modules still work
5. **Update Documentation**: Keep this README current with changes
