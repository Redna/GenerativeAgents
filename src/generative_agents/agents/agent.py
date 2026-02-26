from dataclasses import asdict

# Cognitive Components
from generative_agents.agents.brain import AgentBrain
from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.agents.execution import Execution

from generative_agents.agents.memory.system import MemorySystem
from generative_agents.agents.memory.working import WorkingMemory
from generative_agents.agents.memory.spatial import WorldMap
from generative_agents.common.events import Event, EventType, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.common.models import AgentDTO, MovementDTO
from generative_agents.common.percept import Percept
from generative_agents.persistence.database import initialize_agent
from generative_agents.simulation.maze import Maze, Tile
from generative_agents.simulation.time import DayType, SimulationTime


class Agent:
    def __init__(
        self,
        name: str,
        age: int,
        description: str,
        innate_traits: list[str],
        time: SimulationTime,
        location: str,
        emoji: str,
        activity: str,
        tile: Tile,
        map: WorldMap = None,
    ):
        initialize_agent(name)
        self.name = name
        self.location = location
        self.emoji = emoji
        self.activity = activity
        
        # Working Memory (Short-Term Volatile)
        self.working_memory = WorkingMemory(
            name=name, 
            age=age,
            innate_traits=innate_traits,
            tile=tile, 
            time=time,
            identity_description=description # Initial belief
        )
        
        # World Map (Spatial)
        self.map = WorldMap() if not map else map
        
        # Unified Memory System (Long-Term Semantic/Graph)
        self.memory = MemorySystem(self.name, self.working_memory.retention)
        
        self.time = time
        # Sync time to working memory
        self.working_memory.time = time
        
        # Initialize the Neural Brain
        self.brain = AgentBrain()

        log_agent(self.name, f"Initialized {self.name} at {self.working_memory.tile}")

    def to_dto(self):
        return AgentDTO(
            name=self.name,
            age=self.working_memory.age,
            inniate_traits=self.working_memory.innate_traits,
            description=self.working_memory.identity_description,
            location=self.location,
            emoji=self.emoji,
            activity=self.activity,
            movement=MovementDTO(col=self.working_memory.tile.x, row=self.working_memory.tile.y),
            reflection_trigger_counter=self.working_memory.reflection_trigger_counter,
            reflection_trigger_max=self.working_memory.reflection_trigger_max,
            events_since_last_reflection=self.working_memory.events_since_last_reflection,
        )

    @staticmethod
    def from_dto(dto: AgentDTO, maze: Maze, time: SimulationTime):
        return Agent(
            name=dto.name,
            age=dto.age,
            description=dto.description,
            location=dto.location,
            emoji=dto.emoji,
            innate_traits=dto.inniate_traits,
            activity=dto.activity,
            time=time,
            tile=maze.get_tile(dto.movement.col, dto.movement.row),
        )
        
        # Restore System 2 counters
        agent.working_memory.reflection_trigger_counter = dto.reflection_trigger_counter
        agent.working_memory.reflection_trigger_max = dto.reflection_trigger_max
        agent.working_memory.events_since_last_reflection = dto.events_since_last_reflection
        
        return agent

    @property
    def observation(self):
        if not self.working_memory.action:
            return f"{self.name} is idle"
        else:
            event_description = self.working_memory.action.event.description
            if "(" in event_description:
                event_description = event_description.split("(")[-1][:-1]

            if (
                len(self.working_memory.planned_path) == 0
                and "waiting" not in event_description
            ):
                return f"{self.name} is already {event_description}"

            if "waiting" in event_description:
                return f"{self.name} is {event_description}"

        return f"{self.name} is on the way to {event_description}"

    def perceive(self, percept: Percept):
        """
        Updates the spatial memory based on the incoming percept.
        Note: Cognitive processing of events has moved to AgentBrain.SensoryProcessingLayer.
        """
        for tile in percept.nearby_tiles:
            self.map.add_tile(tile)

    def run_step(
        self,
        percept: Percept,
        maze: Maze,
        agents: dict[str, "Agent"],
        time: SimulationTime,
    ):
        """
        Executes one full cognitive step for the agent using the AgentBrain.
        """
        # Update Time
        daytype: DayType = DayType.SAME_DAY
        if not self.working_memory.daily_schedule:
            daytype = DayType.FIRST_DAY
        elif self.working_memory.time.today != time.today:
            daytype = DayType.NEW_DAY
        self.working_memory.time = time
        self.time = time

        # 1. Perception (Body - Spatial only)
        self.perceive(percept)
        
        # 2. Construct State (Context)
        state = AgentState(
            working_memory=self.working_memory,
            daytype=daytype,
            recent_events=percept.events,
            map=self.map
        )

        # 3. Brain Forward Pass (Reasoning)
        # We pass self.memory.retrieve so the brain's retrieval layer can fetch context
        signal = self.brain(
            percept,
            state,
            retrieve_fn=self.memory.retrieve,
            expand_fn=self.memory.get_context,
            maze=maze,
        )

        # 4. Apply Action Signal (Effectors)
        self._apply_action_signal(signal)

        # Update reflection trigger based on new highly salient memories
        for memory in signal.new_memories:
             if hasattr(memory, 'poignancy') and memory.poignancy >= 0.5:
                 self.working_memory.events_since_last_reflection += 1
                 self.working_memory.reflection_trigger_counter -= memory.poignancy

        # 5. Execution (Motor Control)
        # Convert the decision (Action Address) into movement (Next Tile)
        execution = Execution(self, maze, agents)
        # Use the address from the current action (which might have been updated by the signal)
        target_address = self.working_memory.action.address if self.working_memory.action else f"{self.working_memory.tile.x}:{self.working_memory.tile.y}"
        next_tile = execution.run(target_address)["next_tile"]
        
        # Update tile in working memory
        self.working_memory.tile = next_tile

        # 6. Reflection (System 2 / Offline Learning)
        # Check trigger defined in working memory
        if self.working_memory.reflection_trigger_counter <= 0 and self.working_memory.events_since_last_reflection >= 15:
            from generative_agents.agents.layers.reflection import MemoryConsolidator
            system2 = MemoryConsolidator()
            
            context_events = [
                PerceivedEvent.from_db_entry(m) 
                for m in self.memory.retrieve(state.identity_description, limit=10)
            ]
            
            reflection_state = state 
            reflection_state.recent_events = context_events

            signal_s2 = system2(reflection_state, retrieve_fn=self.memory.retrieve)
            self._apply_action_signal(signal_s2)
            
            # Reset triggers
            self.working_memory.reflection_trigger_counter = self.working_memory.reflection_trigger_max
            self.working_memory.events_since_last_reflection = 0
            
            log_agent(self.name, "System 2 Reflection Completed", "INFO")

        return next_tile

    def _apply_action_signal(self, signal: ActionSignal):
        """
        Applies the output of the brain to the agent's state/memory.
        """
        # 1. Update Plans
        if signal.updated_daily_plan:
            self.working_memory.daily_requirements = signal.updated_daily_plan
        
        if signal.updated_daily_schedule:
            self.working_memory.daily_schedule = signal.updated_daily_schedule
            # Also update hourly organized helper
            self.working_memory.daily_schedule_hourly_organized = [
                (activity, duration) for activity, duration in signal.updated_daily_schedule
            ]

        # 2. Update Memories
        for memory in signal.new_memories:
            self.memory.add(memory)
            
        # 3. Update Action
        if signal.next_action:
            if self.working_memory.action:
                self.working_memory.finished_actions.append(self.working_memory.action)
            self.working_memory.action = signal.next_action

        # 4. Chat State
        if signal.update_chat_buffer:
            self.working_memory.chatting_with_buffer.update(signal.update_chat_buffer)
            
        # 5. Identity Update (System 2)
        if signal.updated_identity:
            self.working_memory.identity_description = signal.updated_identity

