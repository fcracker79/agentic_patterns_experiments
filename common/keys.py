import json
import os
from dataclasses import dataclass

from pydantic import SecretStr

_KEYS_PATH = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'keys.json'))

@dataclass
class Keys:
    openai: SecretStr
    langchain: SecretStr
    tavily: SecretStr

def get_keys() -> Keys:
    with open(_KEYS_PATH, 'r') as f:
        json_data = json.load(f)
    return Keys(**json_data)