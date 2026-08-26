from .base import BaseGenerator
from .prompt import (
    SYSTEM_PROMPT,
    build_qa_prompt,
)
from .validation import (
    validate_grounded_response,
)


def __getattr__(name):
    if name == "OpenAIJSONGenerator":
        from .openai import OpenAIJSONGenerator
        return OpenAIJSONGenerator

    raise AttributeError(name)


__all__ = [
    "BaseGenerator",
    "OpenAIJSONGenerator",
    "SYSTEM_PROMPT",
    "build_qa_prompt",
    "validate_grounded_response",
]
