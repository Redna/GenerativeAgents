from __future__ import annotations

import math
from operator import itemgetter
from typing import TYPE_CHECKING, Dict, List
import concurrent.futures

if TYPE_CHECKING:
    from generative_agents.agents.agent import Agent

from generative_agents.common.models import AgentDTO, RoundUpdateDTO
from generative_agents.common.percept import Percept
from generative_agents.simulation.maze import Level, Maze, Tile
from generative_agents.simulation.time import SimulationTime


class SimulationEngine:
    def __init__(self, maze: Maze, agents: List[Agent], time: SimulationTime):
        self.maze = maze
        self.agents = {agent.name: agent for agent in agents}
        self.time = time
        self.round_updates: List[RoundUpdateDTO] = []

    def step(self):
        """
        Executes one simulation step for all agents concurrently using a ThreadPoolExecutor.
        """
        # 1. Calculate percepts for all agents sequentially
        percepts = self._calculate_percepts()

        # Save old tiles to handle event updates
        old_tiles = {name: agent.working_memory.tile for name, agent in self.agents.items()}

        # 2. Agents decide and execute actions concurrently
        def _run_agent_step(agent, percept, maze, agents, time):
            import asyncio
            async def _run():
                return agent.run_step(percept, maze, agents, time)
            return asyncio.run(_run())

        workers = len(self.agents) if len(self.agents) > 0 else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 20)) as executor:
            futures = {
                executor.submit(_run_agent_step, agent, percepts[agent_name], self.maze, self.agents, self.time): agent_name
                for agent_name, agent in self.agents.items()
            }
            
            for future in concurrent.futures.as_completed(futures):
                agent_name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Agent {agent_name} generated an exception during run_step: {exc}")

        # 3. Update the environment (Tile events) sequentially to avoid race conditions
        self._update_map_events(old_tiles)

        # 4. Record State for API
        self._record_round_update()

        # 5. Advance simulation clock
        self.time.tick()

    def _record_round_update(self):
        agents_dto = [agent.to_dto() for agent in self.agents.values()]
        round_update = RoundUpdateDTO(
            round=len(self.round_updates), time=self.time.as_string(), agents=agents_dto
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
            new_tile = agent.working_memory.tile  # Agent has already moved in run_step

            # Remove old events from old tile
            while agent.working_memory.finished_actions:
                action = agent.working_memory.finished_actions.pop(0)
                if action.event.subject in old_tile.events:
                    del old_tile.events[action.event.subject]

            # Add new event to new tile
            if agent.working_memory.action and agent.working_memory.action.event:
                event = agent.working_memory.action.event
                new_tile.events[event.subject] = event

            # Handle object actions (interactions with objects)
            if agent.working_memory.action and agent.working_memory.action.object_action:
                object_action = agent.working_memory.action.object_action
                if object_action.event:
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
        nearby_tiles = self.maze.get_nearby_tiles(
            agent.working_memory.tile, agent.working_memory.vision_radius
        )

        # Filter for Events
        current_arena = agent.working_memory.tile.get_path(Level.ARENA)
        percept_events_list = []
        percept_events_dict = dict()

        for tile in nearby_tiles:
            if not tile.events:
                continue

            # Agents can only see events in the same arena (room)
            if tile.get_path(Level.ARENA) != current_arena:
                continue

            dist = math.dist(
                [tile.x, tile.y], [agent.working_memory.tile.x, agent.working_memory.tile.y]
            )

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
        for _, event in percept_events_list[: agent.working_memory.attention_bandwidth]:
            visible_events.append(event)

        return Percept(nearby_tiles=nearby_tiles, events=visible_events)
