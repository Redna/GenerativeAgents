from dataclasses import dataclass, asdict
from enum import Enum
from enum import Enum

from generative_agents.common.models import AgentDTO, MovementDTO
from generative_agents.common.events import Event, EventType, PerceivedEvent
from generative_agents.agents.memory.associative import AssociativeMemory
from generative_agents.agents.memory.spatial import MemoryTree
from generative_agents.agents.memory.scratch import Scratch
from generative_agents.common.logging import log_agent
from generative_agents.simulation.maze import Maze, Tile
from generative_agents.persistence.database import initialize_agent
from generative_agents.simulation.time import DayType, SimulationTime
from generative_agents.common.utils import timeit
from generative_agents.common.percept import Percept
from generative_agents.intelligence.poignance import rate_poignance

# Cognitive Components - Direct Imports
from generative_agents.agents.components.execution import Execution
from generative_agents.agents.components.plan import Plan
from generative_agents.agents.components.reflection import Reflection
from generative_agents.agents.components.retrieval import Retrieval

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

        log_agent(self.name, f"Initialized {self.name} at {self.scratch.tile}")

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

    def perceive(self, percept: Percept):
        """
        Processes the incoming percept (what the agent sees/hears).
        updates spatial memory and associative memory.
        """
        # 1. Update Spatial Memory
        for tile in percept.nearby_tiles:
            self.spatial_memory.add(tile)

        # 2. Process Events
        for event in percept.events:
             if not event.predicate:
                event.predicate = "is"

             if not isinstance(event, PerceivedEvent) or event.event_type != EventType.CHAT:
                event = self._perceive_event(event, type_=EventType.EVENT)
                if ":" in event.subject:
                     event.description = f"{event.subject.split(':')[-1]} is {event.description}"

             if event.subject == self.name and event.predicate == "chat with":
                event = self._perceive_event(event, type_=EventType.CHAT)
            
             self.scratch.reflection_trigger_max -= event.poignancy * 10

    def _perceive_event(self, event: Event, type_: EventType = EventType.EVENT):
        if type(event) != PerceivedEvent:
            event_poignancy = self._rate_perception_poignancy(type_, event.description)

            log_agent(self.name, f"event poignancy is {event_poignancy}", "DEBUG")
            event = PerceivedEvent(**asdict(event), event_type=type_, poignancy=event_poignancy)
            event = self.associative_memory.add(event)
        return event

    def _rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = rate_poignance(self.name, self.scratch.identity, event_type.value, description)
        return int(score) / 10

    def run_step(self, percept: Percept, maze: Maze, agents: dict[str, 'Agent'], time: SimulationTime):
        """
        Executes one full cognitive step for the agent.
        Replaces AgentRunner.update.
        """
        # Update Time
        daytype: DayType = DayType.SAME_DAY
        if not self.scratch.time:
            daytype = DayType.FIRST_DAY
        elif (self.scratch.time.today != time.today):
            daytype = DayType.NEW_DAY
        self.scratch.time = time
        self.time = time

        # 1. Perception
        self.perceive(percept)
        perceived_events = percept.events # Used for retrieval

        # 2. Retrieval
        # Retrieve memories relevant to perceived events
        retrieval = Retrieval(self)
        retrieved = retrieval.run(perceived_events)["retrieved"]

        # 3. Plan
        plan = Plan(self, agents)
        # Note: Plan.run expects (daytype, retrieved, focused_event)
        time_today = self.scratch.time.today
        address = plan.run(daytype, retrieved, None)["address"]

        # 4. Execution
        execution = Execution(self, maze, agents)
        next_tile = execution.run(address)["next_tile"]

        # 5. Reflection
        reflection = Reflection(self)
        reflection.run()

        return next_tile