from dataclasses import dataclass
from enum import Enum
from haystack import Pipeline
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

from generative_agents import global_state
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
from generative_agents.utils import timeit

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

        whisper(self.name, f"Initialized {self.name} at {self.scratch.tile}")
    
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


class AgentRunner:
    __TRACER_NAME = "tracer"

    def __init__(self, agent: Agent):
        self.agent = agent
    
    @property
    def pipeline(self):
        pipeline = Pipeline()
        # pipeline.add_component(self.__TRACER_NAME, LangfuseConnector(f"Round {global_state.tick}: {self.agent.name}"))
        pipeline.add_component("perception", Perception(self.agent))
        pipeline.add_component("retrieval", Retrieval(self.agent))
        pipeline.add_component("plan", Plan(self.agent))
        pipeline.add_component("execution", Execution(self.agent))
        pipeline.add_component("reflection", Reflection(self.agent))

        pipeline.connect("perception", "retrieval")
        pipeline.connect("retrieval", "plan")
        pipeline.connect("plan", "execution")
        return pipeline

    @timeit
    def update(self, time: SimulationTime, maze: Maze, agents: dict[str, 'Agent']):
        daytype: DayType = DayType.SAME_DAY

        if not self.agent.scratch.time:
            daytype = daytype.FIRST_DAY
        elif (self.agent.scratch.time.today != time.today):
            daytype = daytype.NEW_DAY

        self.agent.scratch.time = time

        perception = Perception(self.agent)
        retrieval = Retrieval(self.agent)
        plan = Plan(self.agent)
        execution = Execution(self.agent)
        reflection = Reflection(self.agent)


        agent_list = {agent.name: agent for agent in agents}

        perceived = perception.run(maze)["perceived_events"]
        retrieved = retrieval.run(perceived)["retrieved"]
        address = plan.run(agent_list, daytype, retrieved)["address"]
        next_tile = execution.run(maze, agent_list, address)["next_tile"]
        reflection.run()

        #result = self.pipeline.run(
        #            data={"perception": {"maze": maze},
        #                    "plan": {"agents": agents, "daytype": daytype},
        #                    "execution": {"maze": maze, "agents": agents}})

        return next_tile
