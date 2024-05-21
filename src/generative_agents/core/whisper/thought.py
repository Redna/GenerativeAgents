from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel
from generative_agents import global_state

@dataclass
class Thought:
    def __init__(self, agent: str, content: str, level: int):
        self.time = global_state.time.as_string()
        self.tick = global_state.tick
        self.agent = agent
        self.content = content
        self.level = level