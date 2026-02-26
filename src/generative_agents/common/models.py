from typing import List

from pydantic import BaseModel


class MovementDTO(BaseModel):
    col: int
    row: int


class AgentDTO(BaseModel):
    name: str
    age: int
    inniate_traits: List[str] = []
    description: str
    location: str
    emoji: str
    activity: str
    movement: MovementDTO
    reflection_trigger_counter: float = 500.0
    reflection_trigger_max: float = 500.0
    events_since_last_reflection: int = 0


class RoundUpdateDTO(BaseModel):
    round: int
    time: str
    agents: List[AgentDTO]
