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

class RoundUpdateDTO(BaseModel):
    round: int
    time: str
    agents: List[AgentDTO]