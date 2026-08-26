import json
from openai import OpenAI

from .generator_base import BaseGenerator


class OpenAIGenerator(BaseGenerator):

    def __init__(
        self,
        api_key,
        base_url,
        model
    ):

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model


    def generate(
        self,
        prompt: str
    ):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_object"
            }
        )


        content = (
            response
            .choices[0]
            .message
            .content
        )


        return json.loads(
            content
        )