import json
import requests

from .generator_base import BaseGenerator


class OllamaGenerator(BaseGenerator):

    def __init__(
        self,
        model="qwen3.5:9b",
        base_url="http://localhost:11434"
    ):
        self.model = model
        self.url = f"{base_url}/api/generate"


    def generate(
        self,
        prompt: str
    ):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return json.loads(
            result["response"]
        )