from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ):
        """Return a JSON payload plus usage/latency metadata."""
        raise NotImplementedError
