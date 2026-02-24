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
from generative_agents.common.events import Action, Event, EventType
from generative_agents.common.logging import log_agent
from generative_agents.intelligence.modules.environment import WorldPhysicsSimulator
from generative_agents.simulation.dialogue_coordinator import DialogueCoordinator
import asyncio
import random


class SimulationEngine:
    def __init__(self, maze: Maze, agents: List[Agent], time: SimulationTime):
        self.maze = maze
        self.agents = {agent.name: agent for agent in agents}
        self.time = time
        self.round_updates: List[RoundUpdateDTO] = []
        self.physics_simulator = WorldPhysicsSimulator()
        self.dialogue_coordinator = DialogueCoordinator()
        self.is_paused = False
        self.ticks_since_last_save = 0

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

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

        # 3. Intercept and resolve conversations without blocking
        self._process_dialogues()

        # 4. Update the environment (Tile events) sequentially to avoid race conditions
        self._update_map_events(old_tiles)

        # 5. Record State for API
        self._record_round_update()

        # 6. Advance simulation clock
        self.time.tick()

        # 7. Checkpoint Auto-Save
        self.ticks_since_last_save += 1
        if self.ticks_since_last_save >= 100:
            self.save_checkpoint()
            self.ticks_since_last_save = 0

    def save_checkpoint(self):
        import os
        import pickle
        import time
        from generative_agents.common.utils import get_project_root
        
        checkpoints_dir = os.path.join(get_project_root(), "storage", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        filename = f"checkpoint_{int(time.time())}.pkl"
        filepath = os.path.join(checkpoints_dir, filename)
        
        state = {
            "agents": self.agents,
            "time": self.time,
            "round_updates": self.round_updates
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
            
        print(f"Simulation Checkpoint saved to {filepath}")

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

    def get_agent_xray(self, agent_name: str) -> dict | None:
        if agent_name not in self.agents:
            return None
            
        agent = self.agents[agent_name]
        wm = agent.working_memory

        # Fetch recent core memories for overarching context
        try:
            core_memories = agent.memory.retrieve(wm.identity_description, limit=5)
        except Exception:
            core_memories = []
            
        action_dict = None
        if wm.action:
             action_dict = {
                 "address": wm.action.address,
                 "description": wm.action.event.description if wm.action.event else None,
                 "emoji": wm.action.emoji,
                 "start_time": wm.action.start_time.isoformat() if hasattr(wm.action.start_time, "isoformat") else str(wm.action.start_time),
                 "duration": wm.action.duration
             }

        xray = {
            "name": agent.name,
            "time": self.time.as_string(),
            "identity": wm.identity_description,
            "daily_plan": wm.daily_requirements,
            "daily_schedule": wm.daily_schedule_hourly_organized,
            "current_action": action_dict,
            "chatting_with": wm.chatting_with,
            "recent_observations": list(wm.last_observations_cache),
            "core_memories": [m.content for m in core_memories if hasattr(m, 'content')],
            "reflection_trigger_counter": wm.reflection_trigger_counter,
            "reflection_trigger_max": wm.reflection_trigger_max,
        }
        return xray

    def query_agent_memory(self, agent_name: str, query: str, limit: int = 10) -> list[dict]:
        if agent_name not in self.agents:
            return []
            
        agent = self.agents[agent_name]
        memories = agent.memory.retrieve(query, limit=limit)
        
        results = []
        for m in memories:
            entry = {}
            if hasattr(m, "id"): entry["id"] = m.id
            if hasattr(m, "content"): entry["content"] = m.content
            if hasattr(m, "memory_type"): entry["memory_type"] = str(m.memory_type)
            if hasattr(m, "created_at"): entry["created_at"] = str(m.created_at)
            if hasattr(m, "poignancy"): entry["poignancy"] = getattr(m, "poignancy", 0)
            if hasattr(m, "depth"): entry["depth"] = m.depth
            results.append(entry)
            
        return results

    def spawn_agent(self, data: AgentDTO):
        print(f"Spawning agent {data.name} at {data.movement.col}, {data.movement.row}")
        new_agent = Agent.from_dto(data, self.maze, self.time)
        self.agents[new_agent.name] = new_agent

    def _process_dialogues(self):
        """Detects chat initiations, locks states, updates broadcasts, and fires background tasks."""
        for initiator_name, initiator in self.agents.items():
            if initiator.working_memory.chatting_with:
                continue

            action = initiator.working_memory.action
            if action and action.address and action.address.startswith("<persona>"):
                target_name = action.address.split("<persona>")[-1].strip()
                
                if target_name in self.agents:
                    target_agent = self.agents[target_name]
                    
                    if target_agent.working_memory.chatting_with:
                        continue 
                        
                    # A. Lock internal states
                    initiator.working_memory.chatting_with = target_name
                    target_agent.working_memory.chatting_with = initiator_name
                    
                    # B. Update Initiator's broadcast event so they stay in place
                    initiator.working_memory.action.address = "<current>"
                    initiator.working_memory.action.emoji = "💬"
                    initiator.working_memory.action.event.description = f"chatting with {target_name}"

                    # C. Overwrite the Target's current action so bystanders see them talking
                    target_agent.working_memory.action_path_set = False 
                    target_agent.working_memory.planned_path = []
                    target_agent.activity = f"chatting with {initiator_name}"
                    target_agent.working_memory.action = Action(
                        address="<current>",
                        start_time=self.time.time,
                        duration=10, 
                        emoji="💬",
                        event=Event(
                            entity_id=target_name,
                            description=f"chatting with {initiator_name}",
                            tile=target_agent.working_memory.tile,
                            depth=0
                        )
                    )
                    
                    # D. Extract opening line and fire the background task
                    opening_line = "Hello!"
                    for mem in reversed(initiator.working_memory.recent_events): 
                        if target_name in mem.description and "said to" in mem.description:
                            opening_line = mem.description.split(":")[-1].strip()
                            break
                    
                    asyncio.create_task(
                        self.dialogue_coordinator.run_conversation_async(
                            initiator, target_agent, opening_line
                        )
                    )

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
                if action.event.entity_id in old_tile.events:
                    del old_tile.events[action.event.entity_id]
                
                # Also remove old object event if it exists
                if action.object_action and action.object_action.event:
                    obj_event = action.object_action.event
                    if action.object_action.address in self.maze.address_tiles:
                        target_tile = self.maze.address_tiles[action.object_action.address][0]
                        if obj_event.entity_id in target_tile.events:
                            del target_tile.events[obj_event.entity_id]

            # Add new event to new tile
            if agent.working_memory.action and agent.working_memory.action.event:
                event = agent.working_memory.action.event
                new_tile.events[event.entity_id] = event

            # Handle object actions (interactions with objects)
            if agent.working_memory.action and agent.working_memory.action.object_action:
                object_action = agent.working_memory.action.object_action
                if object_action.event:
                    object_event = object_action.event
                    if object_action.address in self.maze.address_tiles:
                        # Note: accessing [0] might be risky if multiple tiles, but follows original logic
                        target_tile = self.maze.address_tiles[object_action.address][0]
                        
                        # 1. Place the standard interaction event on the tile
                        target_tile.events[object_event.entity_id] = object_event
                        
                        # 2. Generate dynamic environmental feedback (with a probability check to avoid API spam)
                        if random.random() < 0.2:
                            action_desc = agent.working_memory.action.event.description
                            object_name = object_event.entity_id
                            
                            consequence_text = self.physics_simulator(
                                agent_name=agent.name, 
                                action_description=action_desc, 
                                object_name=object_name
                            )
                            
                            # 3. Inject the consequence as a new event on the tile
                            if consequence_text:
                                consequence_event = Event(
                                    entity_id=f"Env_{object_name}",
                                    description=consequence_text,
                                    tile=target_tile,
                                    depth=0,
                                )
                                
                                # The agent will perceive this during the NEXT tick's _calculate_percepts
                                target_tile.events[consequence_event.entity_id] = consequence_event
                                log_agent("Environment", f"{consequence_text} (Triggered by {agent.name} interacting with {object_name})", "INFO")

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
                    # Avoid duplicates using the new signature
                    event_sig = (event.entity_id, getattr(event, 'description', ''))
                    if event_sig not in percept_events_dict:
                        percept_events_list.append((dist, event))
                        percept_events_dict[event_sig] = event
            except AttributeError as e:
                print(f"Error reading events from tile: {e}")

        # Sort by distance
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))

        # Apply attention bandwidth
        visible_events = []
        for _, event in percept_events_list[: agent.working_memory.attention_bandwidth]:
            visible_events.append(event)

        return Percept(nearby_tiles=nearby_tiles, events=visible_events)
