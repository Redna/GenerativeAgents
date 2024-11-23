from typing import TypedDict, Annotated

from generative_agents.simulation.time import DayType
from generative_agents.core.events import Event, PerceivedEvent
from generative_agents.persistence.database import ConversationFilling
from generative_agents.simulation.maze import Tile

def upsert(left: dict, right: dict):
    if left is None:
        left = {}
    if right is None:
        right = {}

    left.update(right)
    return left

class AgentRunnerState(TypedDict):
    daytype: DayType
    agent_name: str
    next_tile: str
    perceived_events: list[PerceivedEvent]
    retrieved: list[Event]
    address: str
    focused_event: dict[str, list[PerceivedEvent]]
    next_tile: Tile
    last_conversation: ConversationFilling

class SimulationState(TypedDict):
    simulation_round: int
    agent_states: Annotated[dict[str, AgentRunnerState], upsert]