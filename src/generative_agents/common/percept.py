from dataclasses import dataclass, field
from typing import List

from generative_agents.common.events import Event
from generative_agents.simulation.maze import Tile


@dataclass
class Percept:
    """
    Represents what an agent perceives in a single tick.
    """

    nearby_tiles: List[Tile] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
