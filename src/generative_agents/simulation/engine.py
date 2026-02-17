from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
import math
from operator import itemgetter

if TYPE_CHECKING:
    from generative_agents.agents.agent import Agent

from generative_agents.simulation.maze import Maze, Tile, Level
from generative_agents.common.events import Event, PerceivedEvent
from generative_agents.simulation.time import SimulationTime
from generative_agents.common.percept import Percept


from generative_agents.common.models import AgentDTO, RoundUpdateDTO

class SimulationEngine:
    def __init__(self, maze: Maze, agents: List[Agent], time: SimulationTime):
        self.maze = maze
        self.agents = {agent.name: agent for agent in agents}
        self.time = time
        self.round_updates: List[RoundUpdateDTO] = []

    def step(self):
        """
        Executes one simulation step for all agents.
        """
        # 1. Calculate percepts for all agents
        percepts = self._calculate_percepts()

        # Save old tiles to handle event updates
        old_tiles = {name: agent.scratch.tile for name, agent in self.agents.items()}

        # 2. Agents decide and execute actions
        for agent_name, agent in self.agents.items():
            percept = percepts[agent_name]
            # Calls the new Agent.run_step method
            agent.run_step(percept, self.maze, self.agents, self.time)

        # 3. Update the environment (Tile events)
        self._update_map_events(old_tiles)

        # 4. Record State for API
        self._record_round_update()

    def _record_round_update(self):
        agents_dto = [agent.to_dto() for agent in self.agents.values()]
        round_update = RoundUpdateDTO(
            round=len(self.round_updates),
            time=self.time.as_string(),
            agents=agents_dto
        )
        self.round_updates.append(round_update)

    def get_latest_round_update(self) -> RoundUpdateDTO | None:
        if self.round_updates:
            return self.round_updates[-1]
        return None

    def spawn_agent(self, data: AgentDTO):
        print(f"Spawning agent {data.name} at {data.movement.col}, {data.movement.row}")
        new_agent = Agent.from_dto(data, self.maze, self.time)
        self.agents[new_agent.name] = new_agent

    def _update_map_events(self, old_tiles: Dict[str, Tile]):
        """
        Updates the events on the map tiles based on agent movements and actions.
        Ported from __main__.py _reflect_changes.
        """
        for name, agent in self.agents.items():
            old_tile = old_tiles[name]
            new_tile = agent.scratch.tile # Agent has already moved in run_step

            # Remove old events from old tile
            while agent.scratch.finished_action:
                action = agent.scratch.finished_action.pop(0)
                if action.event.subject in old_tile.events:
                    del old_tile.events[action.event.subject]

            # Add new event to new tile
            event = agent.scratch.action.event
            new_tile.events[event.subject] = agent.scratch.action.event
            
            # Handle object actions (interactions with objects)
            object_action = agent.scratch.action.object_action
            if object_action and object_action.event:
                object_event = object_action.event
                if object_action.address in self.maze.address_tiles:
                    # Note: accessing [0] might be risky if multiple tiles, but follows original logic
                    target_tile = self.maze.address_tiles[object_action.address][0] 
                    target_tile.events[object_event.subject] = object_event
                else:
                    print(f"WARNING: {object_action.address} not in maze")


            # Logging (Optional, but good for debug)
            # print(f"{agent.name} is {agent.emoji} at {new_tile}")

    def _calculate_percepts(self) -> Dict[str, Percept]:
        """
        Calculates the Percept for each agent.
        """
        percepts = {}
        for agent_name, agent in self.agents.items():
            percepts[agent_name] = self.get_percept(agent)
        return percepts

    def get_percept(self, agent: Agent) -> Percept:
        """
        Calculates what a specific agent sees.
        Moves logic from Perception.perceive_space and Perception.perceive_events here.
        """
        nearby_tiles = self.maze.get_nearby_tiles(agent.scratch.tile, agent.scratch.vision_radius)
        
        # Filter for Events
        current_arena = agent.scratch.tile.get_path(Level.ARENA)
        percept_events_list = []
        percept_events_dict = dict()

        for tile in nearby_tiles:
            if not tile.events:
                continue
            
            # Agents can only see events in the same arena (room)
            if tile.get_path(Level.ARENA) != current_arena:
                continue

            dist = math.dist([tile.x, tile.y], [agent.scratch.tile.x, agent.scratch.tile.y])
            
            try:
                for event in tile.events.values():
                    # Avoid duplicates
                    if event.spo_summary not in percept_events_dict:
                        percept_events_list.append((dist, event))
                        percept_events_dict[event.spo_summary] = event
            except Exception as e:
                print(f"Error reading events from tile: {e}")

        # Sort by distance
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))
        
        # Apply attention bandwidth
        visible_events = []
        for _, event in percept_events_list[:agent.scratch.attention_bandwith]:
            visible_events.append(event)

        return Percept(
            nearby_tiles=nearby_tiles,
            events=visible_events
        )
