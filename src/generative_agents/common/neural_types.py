from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from generative_agents.common.events import Action, PerceivedEvent
from generative_agents.simulation.maze import Tile
from generative_agents.simulation.time import SimulationTime, DayType


@dataclass
class AgentState:
    """
    A snapshot of the agent's internal and external state.
    This serves as the 'Context Tensor' for the AgentBrain.
    """
    # Identity
    name: str
    identity_description: str
    innate_traits: List[str]
    
    # Temporal Context
    time: SimulationTime
    daytype: DayType
    
    # Spatial Context
    current_tile: Tile
    
    # Planning Context
    daily_plan_requirements: str
    daily_schedule: List[Tuple[str, int]]
    current_action: Optional[Action]
    
    # Social Context
    chatting_with: Optional[str]
    chatting_with_buffer: Dict[str, int]
    
    # Recent Memory Context (Optional raw feed for the brain)
    # This might be populated by the retrieval layer, but having the top-level
    # summary here is useful.
    recent_events: List[PerceivedEvent] = field(default_factory=list)


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
