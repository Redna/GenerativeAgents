from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from generative_agents.common.events import Action, PerceivedEvent
from generative_agents.simulation.maze import Tile
from generative_agents.simulation.time import SimulationTime, DayType


from generative_agents.agents.memory.working import WorkingMemory
from generative_agents.agents.memory.spatial import WorldMap

@dataclass
class AgentState:
    """
    A snapshot of the agent's internal and external state.
    This serves as the 'Context Tensor' for the AgentBrain.
    """
    working_memory: WorkingMemory
    daytype: DayType
    recent_events: List[PerceivedEvent] = field(default_factory=list)
    map: Optional[WorldMap] = None

    @property
    def name(self) -> str:
        return self.working_memory.name

    @property
    def identity_description(self) -> str:
        return self.working_memory.identity_description

    @property
    def innate_traits(self) -> List[str]:
        return self.working_memory.innate_traits

    @property
    def time(self) -> SimulationTime:
        return self.working_memory.time

    @property
    def current_tile(self) -> Tile:
        return self.working_memory.tile

    @property
    def daily_plan_requirements(self) -> str:
        return self.working_memory.daily_requirements

    @property
    def daily_schedule(self) -> List[Tuple[str, int]]:
        return self.working_memory.daily_schedule

    @property
    def current_action(self) -> Optional[Action]:
        return self.working_memory.action

    @property
    def chatting_with(self) -> Optional[str]:
        return self.working_memory.chatting_with

    @property
    def chatting_with_buffer(self) -> Dict[str, int]:
        return self.working_memory.chatting_with_buffer


@dataclass
class ActionSignal:
    """
    The output of the AgentBrain.
    Contains instructions for the Agent body to execute modification on the world or itself.
    """
    # 1. Primary Action (What to do next)
    next_action: Optional[Action] = None
    
    # 2. Plan Updates (Modifications to the schedule)
    updated_daily_schedule: Optional[List[Tuple[str, int]]] = None
    updated_daily_plan: Optional[str] = None
    
    # 3. Memory Updates (What to write to long-term memory)
    new_memories: list["PerceivedEvent"] = field(default_factory=list)
    updated_identity: str = None # System 2 update for Working Memory
    
    # 4. State Updates
    update_chat_buffer: Optional[Dict[str, int]] = None
    stop_chatting: bool = False
    
    # 5. Metadata for Debugging/Optimization
    thought_trace: str = ""
