from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Generic, TypeVar

from typing_extensions import Self

TCmd = TypeVar("TCmd")


class Device(ABC, Generic[TCmd]):
    """Base class for all devices."""

    def __init__(self, silent: bool = True) -> None:
        """Initialize the device.

        Args:
            silent: If True, suppress output.

        """
        self._silent = silent

    @abstractmethod
    def start(self) -> None:
        """Start the device."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the device."""

    @abstractmethod
    def get_state(self) -> object:
        """Get device input commands."""

    @abstractmethod
    def send_cmd(self, cmd: TCmd) -> None:
        """Send command to device."""

    @contextmanager
    def activate(self) -> Generator[Self, Any, None]:
        """Context manager to activate the device."""
        try:
            self.start()
            yield self
        finally:
            self.stop()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the context manager."""
        self.stop()
        return False
