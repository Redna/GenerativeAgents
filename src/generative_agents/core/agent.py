from enum import Enum
from haystack import Pipeline
from generative_agents.communication.models import AgentDTO, MovementDTO
from generative_agents.core.cognitive_components.execution import Execution
from generative_agents.core.cognitive_components.reflection import Reflection
from generative_agents.core.cognitive_components.perception import Perception
from generative_agents.core.cognitive_components.plan import Plan
from generative_agents.core.cognitive_components.retrieval import Retrieval
from generative_agents.core.memory.associative import AssociativeMemory
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.core.memory.scratch import Scratch
from generative_agents.core.whisper.whisper import whisper
from generative_agents.simulation.maze import Maze, Tile
from generative_agents.persistence.database import initialize_agent
from generative_agents.simulation.time import DayType, SimulationTime


class Agent:
    def __init__(self, name: str, age: int, description: str, innate_traits: list[str], time: SimulationTime, location: str, emoji: str, activity: str, tile: Tile, tree: MemoryTree = None):
        initialize_agent(name)
        self.name = name
        self.location = location
        self.emoji = emoji
        self.activity = activity
        self.scratch = Scratch(name=name, tile=tile, home=tile,
                               innate_traits=innate_traits, age=age)
        self.spatial_memory = MemoryTree() if not tree else tree
        self.associative_memory = AssociativeMemory(
            self.name, self.scratch.retention)
        self.time = time
        self.scratch.tile = tile
        self.scratch.description = description

        self.update_pipeline = Pipeline()
        self.update_pipeline.add_component("perception", Perception(self))
        self.update_pipeline.add_component("retrieval", Retrieval(self))
        self.update_pipeline.add_component("plan", Plan(self))
        self.update_pipeline.add_component("execution", Execution(self))
        self.update_pipeline.add_component("reflection", Reflection(self))

        self.update_pipeline.connect("perception", "retrieval")
        self.update_pipeline.connect("retrieval", "plan")
        self.update_pipeline.connect("plan", "execution")

        whisper(self.name, f"Initialized {self.name} at {self.scratch.tile}")

    def update(self, time: SimulationTime, maze: Maze, agents: dict[str, 'Agent']):
        # random movement
        # if not self.scratch.planned_path:
        #    self.scratch.planned_path = self.__get_random_path(maze)
        #
        # self.agent.tile = self.scratch.planned_path.pop(0)
        # print("Moving to", self.tile)

        daytype: DayType = DayType.SAME_DAY

        if not self.scratch.time:
            daytype = daytype.FIRST_DAY
        elif (self.scratch.time.today != time.today):
            daytype = daytype.NEW_DAY

        self.scratch.time = time

        result = self.update_pipeline.run(
                    data={"perception": {"maze": maze},
                            "plan": {"agents": agents, "daytype": daytype},
                            "execution": {"maze": maze, "agents": agents}})

        return result["execution"]["next_tile"]
    
    def to_dto(self):
        return AgentDTO(
            name=self.name,
            age=self.scratch.age,
            inniate_traits=self.scratch.innate_traits,
            description=self.description,
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
