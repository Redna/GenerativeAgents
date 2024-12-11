from generative_agents.communication.models import AgentDTO, MovementDTO
from generative_agents.core.memory.associative import AssociativeMemory
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.core.memory.scratch import Scratch
from generative_agents.simulation.maze import Maze, Tile
from generative_agents.simulation.time import SimulationTime
from generative_agents.utils import logger

class Agent:
    def __init__(self, name: str, age: int, description: str, innate_traits: list[str], time: SimulationTime, location: str, emoji: str, activity: str, tile: Tile, tree: MemoryTree = None):
        self.name = name
        self.location = location
        self.emoji = emoji
        self.activity = activity
        self.scratch = Scratch(name=name, tile=tile, home=tile,
                               innate_traits=innate_traits, age=age)
        self.spatial_memory = MemoryTree() if not tree else tree
        self.associative_memory = AssociativeMemory(
            self.name, self.scratch.retention)
        self.scratch.time = time
        self.scratch.tile = tile
        self.scratch._identity = ("", description, "")
        self.scratch.description = ""

        logger.log(self.name, f"Initialized {self.name} at {self.scratch.tile}")

    def to_dto(self):
        return AgentDTO(
            name=self.name,
            age=self.scratch.age,
            inniate_traits=self.scratch.innate_traits,
            description=self.scratch.description,
            location=self.location,
            emoji=self.emoji,
            activity=self.activity,
            movement=MovementDTO(col=self.scratch.tile.x,
                                 row=self.scratch.tile.y)
        )

    @staticmethod
    def from_dto(dto: AgentDTO, maze: Maze, time: SimulationTime):
        return Agent(name=dto.name,
                     age=dto.age,
                     description=dto.description,
                     location=dto.location,
                     emoji=dto.emoji,
                     innate_traits=dto.inniate_traits,
                     activity=dto.activity,
                     time=time,
                     tile=maze.get_tile(dto.movement.col, dto.movement.row))

    @property
    def observation(self):
        if not self.scratch.action:
            return f"{self.name} is idle"
        else:
            event_description = self.scratch.action.event.description
            if "(" in event_description:
                event_description = event_description.split("(")[-1][:-1]

            if len(self.scratch.planned_path) == 0 and "waiting" not in event_description:
                return f"{self.name} is already {event_description}"

            if "waiting" in event_description:
                return f"{self.name} is {event_description}"

        return f"{self.name} is on the way to {event_description}"