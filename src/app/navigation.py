from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    icon: str
    enabled: bool
    renderer: Callable | None = None

    @property
    def display_label(
        self,
    ):
        return (
            f"{self.icon} "
            f"{self.label}"
        )
