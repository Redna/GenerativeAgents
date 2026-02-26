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
from generative_agents.common.events import PerceivedEvent
import random


class SimulationEngine:
    def __init__(self, maze: Maze, agents: List[Agent], time: SimulationTime):
        self.maze = maze
        self.agents = {agent.name: agent for agent in agents}
        self.time = time
        self.round_updates: List[RoundUpdateDTO] = []
        self.physics_simulator = WorldPhysicsSimulator()
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
        old_activity = {
            name: (agent.working_memory.action.event.description if agent.working_memory.action and agent.working_memory.action.event else "")
            for name, agent in self.agents.items()
        }

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

        # 3. Detect agents that walked away from a conversation and summarize it
        self._check_conversation_ends(old_activity)

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
            "chatting_with": "", # Extracted dynamically by the frontend if needed
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

    # ------------------------------------------------------------------
    # Emergent Dialogue Lifecycle Management
    # ------------------------------------------------------------------

    def _check_conversation_ends(self, old_activity: Dict[str, str]):
        """
        Detect if an agent was chatting last tick, but is doing something else now.
        If so, the conversation has ended organically. Summarize and inject memory.
        """
        # Dynamic import to avoid circular dependency
        from generative_agents.intelligence.modules.dialogue import DialogueSummarizer, DialoguePlanner
        
        # We only instantiate these if needed this tick
        summarizer = None
        planner = None

        for name, agent in self.agents.items():
            last_act = old_activity.get(name, "")
            
            # Were they chatting last tick?
            if "chatting with" in last_act:
                # Are they STILL chatting this tick?
                current_activity = ""
                if agent.working_memory.action and agent.working_memory.action.event:
                    current_activity = agent.working_memory.action.event.description

                if "chatting with" not in current_activity:
                    # They just walked away!
                    partner_name = last_act.split("chatting with ")[1].split(",")[0].strip()
                    log_agent("Dialogue", f"{agent.name} walked away from conversation with {partner_name}.", "INFO")
                    
                    # Fetch history from memory
                    raw_memories = agent.memory.retrieve(f"conversation with {partner_name}", limit=15)
                    
                    # Filter for recent CHAT events
                    chat_history = []
                    for mem in sorted(raw_memories, key=lambda x: getattr(x, 'created_at', 0)):
                        # Look for 'NAME said to PARTNER:' or 'PARTNER said to NAME:'
                        content = getattr(mem, 'content', str(mem))
                        if "said to" in content and (agent.name in content and partner_name in content):
                             # Format as "Speaker: Utterance"
                             try:
                                 speaker = content.split(" said to ")[0].strip()
                                 utterance = content.split(": ")[1].strip()
                                 chat_history.append(f"{speaker}: {utterance}")
                             except IndexError:
                                 pass

                    if chat_history:
                        if not summarizer:
                            summarizer = DialogueSummarizer()
                            planner = DialoguePlanner()
                            
                        full_chat = "\n".join(chat_history)
                        summary = str(summarizer(full_chat))
                        takeaway = str(planner(agent.name, full_chat))
                        
                        log_agent("Dialogue", f"Conversation summary: {summary}", "INFO")
                        self._inject_conversation_memory(agent, partner_name, summary, takeaway)

    def _inject_conversation_memory(self, agent, partner_name: str, summary: str, takeaway: str):
        """Creates formal memory events from the conversation."""
        summary_event = PerceivedEvent(
            event_type=EventType.CHAT,
            poignancy=0.8,
            depth=1,
            description=f"Had a conversation with {partner_name}. Summary: {summary}",
            entity_id=agent.name,
        )
        agent.memory.add(summary_event)

        takeaway_event = PerceivedEvent(
            event_type=EventType.THOUGHT,
            poignancy=0.7,
            depth=2,
            description=f"After talking to {partner_name}, I need to remember: {takeaway}",
            entity_id=agent.name,
        )
        agent.memory.add(takeaway_event)

    def _update_map_events(self, old_tiles: Dict[str, Tile]):
        """
        Updates the events on the map tiles based on agent movements and actions.
        Ported from __main__.py _reflect_changes.
        """
        # 0. Global Event Garbage Collection (Sweep Expired Environmental quirks)
        for row in self.maze.tiles:
            for tile in row:
                expired_keys = [
                    key for key, event in tile.events.items()
                    if getattr(event, 'expiration', None) and event.expiration <= self.time.time
                ]
                for key in expired_keys:
                    log_agent("Environment", f"Event '{tile.events[key].description}' on {tile} faded away naturally.", "DEBUG")
                    del tile.events[key]

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
                                import datetime
                                # Environmental quirks persist for 30 in-game minutes before fading
                                expiration_time = self.time.time + datetime.timedelta(minutes=30)
                                consequence_event = Event(
                                    entity_id=f"Env_{object_name}_{int(self.time.time.timestamp())}",
                                    description=consequence_text,
                                    tile=target_tile,
                                    depth=0,
                                )
                                # Attach expiration dynamically (duck-typing for the cleanup loop)
                                consequence_event.expiration = expiration_time
                                
                                # The agent will perceive this during the NEXT tick's _calculate_percepts
                                target_tile.events[consequence_event.entity_id] = consequence_event
                                log_agent("Environment", f"{consequence_text} (Triggered by {agent.name} interacting with {object_name})", "INFO")

                    else:
                        print(f"WARNING: {object_action.address} not in maze")

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
