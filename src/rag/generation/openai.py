import json
import time

from openai import OpenAI

from .base import BaseGenerator


class OpenAIJSONGenerator(BaseGenerator):

    def __init__(
        self,
        api_key,
        base_url,
        model,
        input_cost_per_million=None,
        output_cost_per_million=None,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model

        self.input_cost_per_million = (
            input_cost_per_million
        )

        self.output_cost_per_million = (
            output_cost_per_million
        )


    def _estimate_cost(
        self,
        prompt_tokens,
        completion_tokens,
    ):
        if (
            self.input_cost_per_million
            is None
            or self.output_cost_per_million
            is None
        ):
            return None

        input_cost = (
            prompt_tokens
            / 1_000_000
            * self.input_cost_per_million
        )

        output_cost = (
            completion_tokens
            / 1_000_000
            * self.output_cost_per_million
        )

        return float(
            input_cost + output_cost
        )


    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ):
        start = time.perf_counter()

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_object"
                },
            )
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        content = (
            response
            .choices[0]
            .message
            .content
        )

        payload = json.loads(
            content
        )

        usage = getattr(
            response,
            "usage",
            None,
        )

        prompt_tokens = int(
            getattr(
                usage,
                "prompt_tokens",
                0,
            )
            or 0
        )

        completion_tokens = int(
            getattr(
                usage,
                "completion_tokens",
                0,
            )
            or 0
        )

        total_tokens = int(
            getattr(
                usage,
                "total_tokens",
                prompt_tokens
                + completion_tokens,
            )
            or (
                prompt_tokens
                + completion_tokens
            )
        )

        estimated_cost_usd = (
            self._estimate_cost(
                prompt_tokens,
                completion_tokens,
            )
        )

        return {
            "payload": payload,
            "model": self.model,
            "latency_ms": float(
                latency_ms
            ),
            "prompt_tokens": (
                prompt_tokens
            ),
            "completion_tokens": (
                completion_tokens
            ),
            "total_tokens": (
                total_tokens
            ),
            "estimated_cost_usd": (
                estimated_cost_usd
            ),
        }
