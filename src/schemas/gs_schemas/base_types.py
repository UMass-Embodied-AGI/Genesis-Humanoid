import enum
from typing import Any

from pydantic import BaseModel, ConfigDict

# -----------------------------------------------------------------------------
# Pydantic
# -----------------------------------------------------------------------------


def genesis_pydantic_config(
    frozen: bool = False,
    arbitrary_types_allowed: bool = False,
) -> ConfigDict:
    """
    Default config for all Genesis Pydantic dataclasses and models
    https://docs.pydantic.dev/latest/api/config

    Usage:
        from pydantic.dataclasses import dataclass

        @dataclass(config=genesis_pydantic_config())
        class MyDataClass():
            ...
    """
    return ConfigDict(
        # forbid extra fields not defined in the model
        extra="forbid",
        # validate default values against field types
        validate_default=True,
        # don't protect any namespaces from field names
        protected_namespaces=(),
        # serialize enums using their values
        use_enum_values=True,
        # freeze the dataclass
        frozen=frozen,
        # allow arbitrary types (e.g., np.ndarray, torch.Tensor)
        arbitrary_types_allowed=arbitrary_types_allowed,
    )


class GenesisPydanticModel(BaseModel):
    """
    Base model class for Genesis Pydantic models with consistent configuration.

    Usage:
        class MyModel(GenesisPydanticModel):
            field1: str
            field2: int
    """

    model_config = genesis_pydantic_config()


# -----------------------------------------------------------------------------
# Enum
# -----------------------------------------------------------------------------


class GenesisEnum(str, enum.Enum):
    """
    Base enum class for Genesis enums. Enforces that enum values are strings and match
    their keys and enables direct string comparison with enum values.

    Usage:
        class MyEnum(GenesisEnum):
            FOO = "FOO"
            BAR = "BAR"
    """

    def __init_subclass__(cls, **kwargs) -> None:
        # ensure values are unique and match with names
        unique_values = set()
        for name, member in cls.__members__.items():
            if member.value in unique_values:
                raise ValueError(f"Duplicate value found in {cls.__name__}: {member.value}")
            if name != member.value:
                raise ValueError(
                    f"Member name must match its value in {cls.__name__}: '{name}' != '{member.value}'"
                )
            unique_values.add(member.value)
        super().__init_subclass__(**kwargs)

    def __eq__(self, other: Any) -> bool:
        # allow comparison with string values directly
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
