"""
agents/layers/actor.py
Phase 6: ActorLayer — single ReActActor call replaces 7+ sequential LLM decision calls.
The model picks a tool; Python dispatches it.
"""
import datetime
import dspy
from typing import Optional, Tuple, List

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import Action, Event, EventType, ObjectAction, PerceivedEvent
from generative_agents.common.logging import log_agent

from generative_agents.intelligence.modules.actor_react import ReActActor, ToolCall
from generative_agents.intelligence.modules.perception import EventParser, heuristic_emoji


class ActorLayer(dspy.Module):
    """
    Phase 6: One ReActActor LLM call → ToolCall → Python dispatch.
    Pure-Python helpers (schedule index, action-finished check) are unchanged.
    """

    def __init__(self):
        super().__init__()
        self.react_actor = ReActActor()
        self.event_parser = EventParser()

    # ------------------------------------------------------------------
    # Main forward pass — ONE LLM call
    # ------------------------------------------------------------------

    def forward(self, state: AgentState, plan_signal: ActionSignal) -> ActionSignal:
        """
        Returns an ActionSignal based on a single ReActActor tool-choice call.
        """
        schedule = plan_signal.updated_daily_schedule if plan_signal.updated_daily_schedule else state.daily_schedule

        # Skip cognition if current action is still running
        if not self._is_action_finished(state):
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

        return self._dispatch(tool_call, state, current_plan)

    # ------------------------------------------------------------------
    # Tool dispatch (pure Python — zero extra LLM calls)
    # ------------------------------------------------------------------

    def _dispatch(self, tool_call: ToolCall, state: AgentState, current_plan: str) -> ActionSignal:
        t = tool_call.tool
        args = tool_call.args

        if t == "speak_to":
            return self._dispatch_speak(state, args)
        elif t == "move_to":
            return self._dispatch_move(state, args, current_plan)
        elif t == "update_action":
            return self._dispatch_update_action(state, args)
        else:  # "wait"
            return ActionSignal()  # no-op: current action continues

    def _dispatch_move(self, state: AgentState, args: dict, fallback_plan: str) -> ActionSignal:
        destination = args.get("destination", fallback_plan) or fallback_plan
        subject, predicate, object_ = self.event_parser.get_triple(state.name, destination)
        action = Action(
            address=destination,
            start_time=state.time.time,
            duration=60,
            emoji=heuristic_emoji(destination),
            event=Event(
                subject=subject,
                predicate=predicate,
                object_=object_,
                description=destination,
                tile=state.current_tile,
                depth=0,
            ),
        )
        return ActionSignal(next_action=action)

    def _dispatch_speak(self, state: AgentState, args: dict) -> ActionSignal:
        target = args.get("target_agent", "")
        line = args.get("opening_line", "Hello.")

        if not target or state.chatting_with or state.chatting_with_buffer.get(target, 0) > 0:
            return ActionSignal()

        subject, predicate, object_ = self.event_parser.get_triple(state.name, f"chat with {target}")
        chat_action = Action(
            address=f"<persona> {target}",
            start_time=state.time.time,
            duration=10,
            emoji="💬",
            event=Event(
                subject=subject,
                predicate=predicate,
                object_=object_,
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
            subject=state.name,
            predicate="said",
            object_=target,
            created=state.time.time,
            expiration=state.time.time + datetime.timedelta(days=7),
            tile=state.current_tile,
        )
        sig.new_memories.append(chat_event)
        return sig

    def _dispatch_update_action(self, state: AgentState, args: dict) -> ActionSignal:
        activity = args.get("activity", "idle")
        subject, predicate, object_ = self.event_parser.get_triple(state.name, activity)
        action = Action(
            address="<current>",
            start_time=state.time.time,
            duration=60,
            emoji=heuristic_emoji(activity),
            event=Event(
                subject=subject,
                predicate=predicate,
                object_=object_,
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
            duration_int = 60
        return state.time.time >= start + datetime.timedelta(minutes=duration_int)

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
