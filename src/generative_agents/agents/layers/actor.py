"""
agents/layers/actor.py
Phase 6: ActorLayer — single ReActActor call replaces 7+ sequential LLM decision calls.
The model picks a tool; Python dispatches it.
"""
import datetime
import random
import dspy
import difflib
import numpy as np
from typing import Optional, Tuple, List, Dict

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import Action, Event, EventType, ObjectAction, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.simulation.maze import Maze, Level

from generative_agents.intelligence.modules.perception import heuristic_emoji

# How poignant an event must be to interrupt a running action
INTERRUPT_POIGNANCY_THRESHOLD = 0.7
# Default action duration in minutes (shorter = more responsive agents)
DEFAULT_ACTION_DURATION = 10



class AgentStepSignature(dspy.Signature):
    """
    You are a generative agent deciding your next action for this simulation tick.
    Look at what you can see around you (visible_events) and your current daily plan.
    Choose EXACTLY ONE tool from the available tools to advance your goals.
    """
    name: str = dspy.InputField(desc="Agent's name.")
    identity: str = dspy.InputField(desc="Agent's identity, backstory, and traits.")
    current_plan: str = dspy.InputField(desc="The agent's current scheduled activity.")
    visible_events: str = dspy.InputField(desc="Numbered list of events the agent can perceive right now.")
    
    tools: list[dspy.Tool] = dspy.InputField(desc="Available tools to call.")
    outputs: dspy.ToolCalls = dspy.OutputField(desc="The tool call to execute.")


class ActorLayer(dspy.Module):
    """
    Phase 6: One native dspy.Predict call returning dspy.ToolCalls → Python dispatch.
    Pure-Python helpers (schedule index, action-finished check) are unchanged.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AgentStepSignature)
        self._location_embeddings_cache: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Helper to fetch vectors via DSPy's configured embedding model."""
        try:
            from generative_agents.agents.memory.repository import VLLMEmbedder
            embedder = VLLMEmbedder()
            result = embedder.encode([text])[0]
            return np.array(result) if result else np.zeros(1)
        except Exception as e:
            # Fallback if embedding generation fails
            log_agent("System", f"Embedding generation failed: {e}", "WARNING")
            return np.zeros(1)

    def _semantic_search_location(self, target: str, candidates: List[str]) -> Optional[str]:
        """Find the semantically closest location using cosine similarity."""
        target_emb = self._get_embedding(target)
        if not target_emb.any():
            return None

        best_score = -1.0
        best_match = None

        for candidate in candidates:
            if candidate not in self._location_embeddings_cache:
                self._location_embeddings_cache[candidate] = self._get_embedding(candidate)
            
            cand_emb = self._location_embeddings_cache[candidate]
            if not cand_emb.any(): continue

            # Cosine similarity
            score = np.dot(target_emb, cand_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(cand_emb) + 1e-10)
            if score > best_score:
                best_score = score
                best_match = candidate

        # Threshold to avoid hallucinating a totally unrelated location
        if best_score > 0.65:
            return best_match
        return None

    # ------------------------------------------------------------------
    # Dynamic Tool Definition
    # ------------------------------------------------------------------

    def _build_tools(self, state: AgentState, current_plan: str, maze: Maze = None) -> List[dspy.Tool]:
        def explore(strategy: str) -> ActionSignal:
            """Use this when you need to go somewhere but DO NOT know the location (e.g. if a move_to fails or you are lost). Strategies: 'wander randomly', 'search for [item/person]', or 'map area'."""
            action = Action(
                address="<random>",
                start_time=state.time.time,
                duration=DEFAULT_ACTION_DURATION,
                emoji="🗺️",
                event=Event(
                    entity_id=state.name,
                    description=f"exploring the area by {strategy}",
                    tile=state.current_tile,
                    depth=0,
                ),
            )
            return ActionSignal(next_action=action)

        def move_to(action_description: str, target_location: str) -> ActionSignal:
            """PRIMARY tool for starting any new physical activity. 
            Args:
                action_description: Natural language description of what you are doing (e.g. 'Making breakfast', 'Working on laptop').
                target_location: The specific location or room you want to go to (e.g. 'Hobbs Cafe', 'kitchen', 'desk').
            """
            resolved_address = self._resolve_address(target_location, state, maze)
            log_agent(state.name, f"Resolved target location '{target_location}' → '{resolved_address}'", "DEBUG")

            if resolved_address == "UNKNOWN_LOCATION":
                return explore(f"searching for {target_location}")

            action = Action(
                address=resolved_address,
                start_time=state.time.time,
                duration=DEFAULT_ACTION_DURATION,
                emoji=heuristic_emoji(action_description),
                event=Event(
                    entity_id=state.name,
                    description=action_description,
                    tile=state.current_tile,
                    depth=0,
                ),
            )

            if resolved_address and resolved_address.count(":") >= 3:
                object_name = resolved_address.split(":")[-1]
                action.object_action = ObjectAction(
                    address=resolved_address,
                    emoji="⚡",
                    event=Event(
                        entity_id=object_name,
                        description=f"is in use by {state.name}",
                        depth=0
                    )
                )
            return ActionSignal(next_action=action)

        def speak_to(target_agent: str, opening_line: str) -> ActionSignal:
            """Initiate dialogue with a nearby agent."""
            # DO NOT block if already chatting — this is how they reply!
            if not target_agent or state.chatting_with_buffer.get(target_agent, 0) > 0:
                return ActionSignal()

            # Validate target_agent exists in perceived events to prevent hallucinatory conversations
            perceivable_entities = {str(e.entity_id) for e in state.recent_events if e.entity_id != state.name}
            
            if target_agent not in perceivable_entities and target_agent not in state.chatting_with_buffer:
                # Try a lexical fuzzy match in case the LLM slightly misspelled their name
                lexical_matches = difflib.get_close_matches(target_agent, perceivable_entities, n=1, cutoff=0.6)
                if lexical_matches:
                    log_agent(state.name, f"Corrected dialogue target '{target_agent}' -> '{lexical_matches[0]}'", "DEBUG")
                    target_agent = lexical_matches[0]
                else:
                    log_agent(state.name, f"Cannot speak to '{target_agent}': No such agent perceived nearby.", "WARNING")
                    return wait()

            chat_action = Action(
                address=f"<persona> {target_agent}",
                start_time=state.time.time,
                duration=10,
                emoji="💬",
                event=Event(
                    entity_id=state.name,
                    description=f"chatting with {target_agent}",
                    tile=state.current_tile,
                    depth=0,
                ),
            )
            sig = ActionSignal(next_action=chat_action)
            sig.update_chat_buffer = {target_agent: 60}

            chat_event = PerceivedEvent(
                event_type=EventType.CHAT,
                poignancy=0.6,
                depth=1,
                description=f"{state.name} said to {target_agent}: {opening_line}",
                entity_id=state.name,
                created=state.time.time,
                expiration=state.time.time + datetime.timedelta(days=7),
                tile=state.current_tile,
            )
            sig.new_memories.append(chat_event)
            return sig

        def update_action(activity: str) -> ActionSignal:
            """ONLY use this if you are changing your state but STAYING EXACTLY WHERE YOU ARE (e.g. 'Reading a book', 'Using computer'). If your new activity requires interacting with a new object or room, use move_to instead!"""
            action = Action(
                address="<current>",
                start_time=state.time.time,
                duration=DEFAULT_ACTION_DURATION,
                emoji=heuristic_emoji(activity),
                event=Event(
                    entity_id=state.name,
                    description=activity,
                    tile=state.current_tile,
                    depth=0,
                ),
            )
            return ActionSignal(next_action=action)

        def wait() -> ActionSignal:
            """Do nothing. Stay in place and let your current action continue."""
            return update_action(current_plan)

        return [
            dspy.Tool(move_to),
            dspy.Tool(speak_to),
            dspy.Tool(wait),
            dspy.Tool(update_action),
            dspy.Tool(explore),
        ]

    # ------------------------------------------------------------------
    # Main forward pass — ONE LLM call
    # ------------------------------------------------------------------

    def forward(self, state: AgentState, plan_signal: ActionSignal, maze: Maze = None) -> ActionSignal:
        """
        Returns an ActionSignal based on a single native tool-choice call.
        """
        schedule = plan_signal.updated_daily_schedule if plan_signal.updated_daily_schedule else state.daily_schedule

        # Skip cognition unless action is finished OR a highly poignant event interrupts
        if not self._should_act(state):
            return ActionSignal()

        if not schedule:
            log_agent(state.name, "ActorLayer: No schedule available", "WARNING")
            return ActionSignal()

        # Build context for the model
        current_plan = self._current_schedule_item(state, schedule)
        events_str = "\n".join(f"{i + 1}. {e.description}" for i, e in enumerate(state.recent_events)) or "Nothing notable nearby."

        # Dynamically build tools injected with current state
        tools = self._build_tools(state, current_plan, maze)

        # --- SINGLE LLM CALL ---
        try:
            result = self.predict(
                name=state.name,
                identity=state.identity_description,
                current_plan=current_plan,
                visible_events=events_str,
                tools=tools,
            )
            
            if hasattr(result, "outputs") and hasattr(result.outputs, "tool_calls") and len(result.outputs.tool_calls) > 0:
                 tool_call = result.outputs.tool_calls[0]
                 log_agent(state.name, f"ActorLayer: tool={tool_call.name} args={tool_call.args}", "INFO")
                 
                 # Dynamically execute chosen tool natively against the Python closure!
                 return tool_call.execute(functions=tools)
            else:
                 log_agent(state.name, f"ActorLayer: LLM provided no tool calls. Defaulting to wait.", "WARNING")
                 return tools[2]() # fail-safe wait()

        except Exception as e:
            log_agent(state.name, f"ActorLayer Exception: {e}", "ERROR")
            return tools[2]() # fail-safe wait()

    # ------------------------------------------------------------------
    # Pure-Python helpers (no LLM)
    # ------------------------------------------------------------------

    def _is_action_finished(self, state: AgentState) -> bool:
        if not state.current_action:
            return True
        start = state.current_action.start_time
        if not start:
            return True
        if start.second != 0:
            start = start.replace(second=0) + datetime.timedelta(minutes=1)
        try:
            duration_int = int(state.current_action.duration)
        except (ValueError, TypeError):
            duration_int = DEFAULT_ACTION_DURATION
        return state.time.time >= start + datetime.timedelta(minutes=duration_int)

    def _should_act(self, state: AgentState) -> bool:
        """Returns True if the agent should make a new decision this tick."""
        # Re-evaluate every tick while chatting to allow replies
        if state.current_action and state.current_action.event and "chatting with" in state.current_action.event.description:
            return True
            
        if self._is_action_finished(state):
            return True
        # Interrupt if any recent event is highly poignant
        for event in state.recent_events:
            if hasattr(event, 'poignancy') and event.poignancy >= INTERRUPT_POIGNANCY_THRESHOLD:
                log_agent(state.name, f"Interrupting action for: {event.description} (poignancy={event.poignancy})", "INFO")
                return True
        return False

    def _resolve_address(self, target: str, state: AgentState, maze: Maze = None) -> str:
        """Resolve destination against the agent's INTERNAL memory, not the global maze."""
        if not state.map:
            return target
            
        # 1. Gather all known node paths (e.g. "the Ville:Hobbs Cafe:cafe:cafe customer seating")
        known_locations = list(state.map.known_addresses)
        
        # 2. Extract base names for easier matching (e.g. "cafe customer seating")
        basenames = {addr.split(":")[-1]: addr for addr in known_locations if ":" in addr}
        for addr in known_locations: 
            if addr not in basenames.values(): basenames[addr] = addr

        # Tier 1: Exact Match
        if target in known_locations:
            return target
        if target in basenames:
            return basenames[target]

        # Tier 2: Lexical Fuzzy Match (Levenshtein / Difflib) - e.g. "Hobs Cafe" -> "Hobbs Cafe"
        # We try matching against the basenames first (more natural for typos)
        lexical_matches = difflib.get_close_matches(target, basenames.keys(), n=1, cutoff=0.5)
        if lexical_matches:
            log_agent(state.name, f"Lexical fuzzy match '{target}' -> '{lexical_matches[0]}'", "DEBUG")
            return basenames[lexical_matches[0]]

        # Tier 3: Semantic Search (Embeddings) - e.g. "grocery store" -> "The Willows Market and Pharmacy"
        semantic_match = self._semantic_search_location(target, list(basenames.keys()))
        if semantic_match:
            log_agent(state.name, f"Semantic embedding match '{target}' -> '{semantic_match}'", "DEBUG")
            return basenames[semantic_match]

        # If all fail, the agent is spatially ignorant of this target
        log_agent(state.name, f"Location '{target}' is unknown to the agent lexically and semantically.", "WARNING")
        return "UNKNOWN_LOCATION"

    def _current_schedule_item(self, state: AgentState, schedule: List[Tuple[str, int]]) -> str:
        """Returns the description of the current schedule slot."""
        if not schedule:
            return "idle"
        # Find schedule slot matching current sim hour
        hour = state.time.time.hour
        elapsed_hours = 0
        for activity, duration_min in schedule:
            slot_hours = max(1, duration_min // 60)
            if elapsed_hours + slot_hours > hour:
                return activity
            elapsed_hours += slot_hours
        return schedule[-1][0] if schedule else "idle"
