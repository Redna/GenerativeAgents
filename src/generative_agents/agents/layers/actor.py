"""
agents/layers/actor.py
Phase 6: ActorLayer — single ReActActor call replaces 7+ sequential LLM decision calls.
The model picks a tool; Python dispatches it.
"""
import datetime
import random
import dspy
from typing import Optional, Tuple, List

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import Action, Event, EventType, ObjectAction, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.simulation.maze import Maze

from generative_agents.intelligence.modules.actor_react import ReActActor, ToolCall
from generative_agents.intelligence.modules.perception import heuristic_emoji

# How poignant an event must be to interrupt a running action
INTERRUPT_POIGNANCY_THRESHOLD = 0.7
# Default action duration in minutes (shorter = more responsive agents)
DEFAULT_ACTION_DURATION = 10

class ActorLayer(dspy.Module):
    """
    Phase 6: One ReActActor LLM call → ToolCall → Python dispatch.
    Pure-Python helpers (schedule index, action-finished check) are unchanged.
    """

    def __init__(self):
        super().__init__()
        super().__init__()
        self.react_actor = ReActActor()

    # ------------------------------------------------------------------
    # Main forward pass — ONE LLM call
    # ------------------------------------------------------------------

    def forward(self, state: AgentState, plan_signal: ActionSignal, maze: Maze = None) -> ActionSignal:
        """
        Returns an ActionSignal based on a single ReActActor tool-choice call.
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
        visible_events = [e.description for e in state.recent_events]

        # --- SINGLE LLM CALL ---
        tool_call: ToolCall = self.react_actor(
            name=state.name,
            identity=state.identity_description,
            current_plan=current_plan,
            visible_events=visible_events,
        )
        log_agent(state.name, f"ActorLayer: tool={tool_call.tool} args={tool_call.args}", "INFO")

        return self._dispatch(tool_call, state, current_plan, maze)

    # ------------------------------------------------------------------
    # Tool dispatch (pure Python — zero extra LLM calls)
    # ------------------------------------------------------------------

    def _dispatch(self, tool_call: ToolCall, state: AgentState, current_plan: str, maze: Maze = None) -> ActionSignal:
        t = tool_call.tool
        args = tool_call.args

        if t == "speak_to":
            return self._dispatch_speak(state, args)
        elif t == "move_to":
            return self._dispatch_move(state, args, current_plan, maze)
        elif t == "update_action":
            return self._dispatch_update_action(state, args)
        else:  # "wait"
            # Stay in place and continue current scheduled activity
            return self._dispatch_update_action(state, {"activity": current_plan})

    def _dispatch_move(self, state: AgentState, args: dict, fallback_plan: str, maze: Maze = None) -> ActionSignal:
        destination = args.get("action_description", fallback_plan) or fallback_plan

        # Resolve natural-language destination to a valid maze address
        resolved_address = self._resolve_address(destination, maze)
        log_agent(state.name, f"Resolved '{destination}' → '{resolved_address}'", "DEBUG")

        action = Action(
            address=resolved_address,
            start_time=state.time.time,
            duration=DEFAULT_ACTION_DURATION,
            emoji=heuristic_emoji(destination),
            event=Event(
                entity_id=state.name,
                description=destination,
                tile=state.current_tile,
                depth=0,
            ),
        )

        # Detect if destination is a Game Object (World:Sector:Arena:Object format)
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

    def _dispatch_speak(self, state: AgentState, args: dict) -> ActionSignal:
        target = args.get("target_agent", "")
        line = args.get("opening_line", "Hello.")

        if not target or state.chatting_with or state.chatting_with_buffer.get(target, 0) > 0:
            return ActionSignal()

        chat_action = Action(
            address=f"<persona> {target}",
            start_time=state.time.time,
            duration=10,
            emoji="💬",
            event=Event(
                entity_id=state.name,
                description=f"chatting with {target}",
                tile=state.current_tile,
                depth=0,
            ),
        )
        sig = ActionSignal(next_action=chat_action)
        sig.update_chat_buffer = {target: 60}

        # Store opening line as a memory
        chat_event = PerceivedEvent(
            event_type=EventType.CHAT,
            poignancy=0.6,
            depth=1,
            description=f"{state.name} said to {target}: {line}",
            entity_id=state.name,
            created=state.time.time,
            expiration=state.time.time + datetime.timedelta(days=7),
            tile=state.current_tile,
        )
        sig.new_memories.append(chat_event)
        return sig

    def _dispatch_update_action(self, state: AgentState, args: dict) -> ActionSignal:
        activity = args.get("activity", "idle")
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
        if self._is_action_finished(state):
            return True
        # Interrupt if any recent event is highly poignant
        for event in state.recent_events:
            if hasattr(event, 'poignancy') and event.poignancy >= INTERRUPT_POIGNANCY_THRESHOLD:
                log_agent(state.name, f"Interrupting action for: {event.description} (poignancy={event.poignancy})", "INFO")
                return True
        return False

    def _resolve_address(self, activity: str, maze: Maze = None) -> str:
        """Resolve a natural-language activity to a valid maze address via fuzzy matching."""
        if maze is None:
            return activity
        # Exact match first
        if activity in maze.address_tiles:
            return activity
        # Fuzzy match: find addresses containing the activity string (case-insensitive)
        matches = {
            addr: tiles
            for addr, tiles in maze.address_tiles.items()
            if activity.lower() in addr.lower()
        }
        if matches:
            # Pick the shortest matching address (most specific)
            best = min(matches.keys(), key=len)
            return best
        # Try matching individual words from the activity
        words = [w for w in activity.lower().split() if len(w) > 3]
        for word in words:
            word_matches = {
                addr: tiles
                for addr, tiles in maze.address_tiles.items()
                if word in addr.lower()
            }
            if word_matches:
                best = min(word_matches.keys(), key=len)
                return best
        # No match — pick a random known address
        log_agent("System", f"No maze address found for '{activity}', using random", "WARNING")
        return random.choice(list(maze.address_tiles.keys()))

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
