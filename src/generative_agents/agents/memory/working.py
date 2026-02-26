import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from generative_agents.simulation.time import SimulationTime
from generative_agents.simulation.maze import Tile
from generative_agents.common.events import Action

@dataclass
class WorkingMemory:
    """
    Holds the agent's volatile working memory (Short-Term).
    Replaces the legacy 'Scratch' component.
    """
    # Identity & Traits (Beliefs)
    name: str
    age: int
    innate_traits: List[str]
    learned_traits: List[str] = field(default_factory=list)
    lifestyle: str = ""
    identity_description: str = "" # The core "I am..." belief, updated by System 2
    
    # Spatio-Temporal Context
    tile: Tile = None
    time: Optional[SimulationTime] = None
    
    # Current Goal/Action State
    daily_requirements: str = ""
    daily_schedule: List[Tuple[str, int]] = field(default_factory=list)
    daily_schedule_hourly_organized: List[Tuple[str, int]] = field(default_factory=list)
    
    action: Optional[Action] = None
    action_path_set: bool = False
    planned_path: List[Tile] = field(default_factory=list)
    finished_actions: List[Action] = field(default_factory=list)
    
    # Social Context
    chatting_with_buffer: Dict[str, int] = field(default_factory=dict)  # cooldown tracker
    
    # Metadata
    retention: int = 5
    vision_radius: int = 6
    attention_bandwidth: int = 4
    last_observations_cache: set[str] = field(default_factory=set)
    repeated_event_fade: Dict[str, float] = field(default_factory=dict) # Tracks decay factor for repetitive events
    
    # System 2 Triggers
    reflection_trigger_counter: int = 500  # Starts higher to allow for natural day progression
    reflection_trigger_max: int = 500
    events_since_last_reflection: int = 0  # Starts at 0 instead of 800 to prevent immediate first-day trigger
    
    def is_action_finished(self) -> bool:
        """
        Checks if current action is finished based on time.
        """
        if not self.action:
            return True

        start = self.action.start_time
        # Adjust for 0 seconds if needed, legacy logic preserved
        if start.second != 0:
            start = start.replace(second=0) + datetime.timedelta(minutes=1)
        end_time = start + datetime.timedelta(minutes=self.action.duration)

        if end_time and self.time.time >= end_time:
            return True
            
        return False
