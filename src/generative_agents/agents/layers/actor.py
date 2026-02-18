import datetime
import dspy
from typing import Optional, Tuple, List

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import Action, Event, EventType, ObjectAction, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.simulation.maze import Level

# Intelligence Functions
from generative_agents.intelligence.modules.perception import EmojiMapper, EventParser, ReactionDecider
from generative_agents.intelligence.modules.dialogue import (
    TalkDecider, DialogueGenerator, DialogueSummarizer, DialogueMemoer
)
from generative_agents.intelligence.modules.spatial import (
    SectorSelector, ArenaSelector, ObjectSelector
)

class ActorLayer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Perception
        self.emoji_mapper = EmojiMapper()
        self.event_parser = EventParser()
        self.reaction_decider = ReactionDecider()
        
        # Dialogue
        self.talk_decider = TalkDecider()
        self.dialogue_generator = DialogueGenerator()
        self.dialogue_summarizer = DialogueSummarizer()

        # Spatial
        self.sector_selector = SectorSelector()
        self.arena_selector = ArenaSelector()
        self.object_selector = ObjectSelector()
        
    def forward(self, state: AgentState, plan_signal: ActionSignal) -> ActionSignal:
        """
        Determines the immediate next action.
        Integrates logic from Plan._determine_action and Plan._react.
        """
        # If PlanningLayer produced a signal (e.g. new schedule), we should respect it
        schedule = plan_signal.updated_daily_schedule if plan_signal.updated_daily_schedule else state.daily_schedule
        
        # 1. Check if current action is finished
        if not self._is_action_finished(state):
             # If action is not finished, we generally don't change anything, 
             # unless there's a strong reaction trigger (not implemented yet for interruption)
             return ActionSignal() 
             
        # Guard clause if no schedule exists at all (e.g. error in planning)
        if not schedule:
             log_agent(state.name, "No schedule available for ActorLayer", "WARNING")
             return ActionSignal()

        # 2. Determine Next Action from Schedule
        signal = self._determine_next_action_from_schedule(state, schedule)
        
        # 3. Check for Reactions (Chat, Wait, etc)
        # We only check for reaction if we are not already chatting or in high priority action
        if not state.current_action or "sleeping" not in state.current_action.event.description:
             reaction_signal = self._check_reaction(state, signal)
             if reaction_signal: 
                 return reaction_signal
        
        return signal

    def _check_reaction(self, state: AgentState, planned_signal: ActionSignal) -> Optional[ActionSignal]:
        """
        Checks if the agent should react to nearby events (specifically other agents).
        """
        # 1. Focus finding (simplified: look for other agents in nearby tiles)
        # Note: state.recent_events contains perceived events.
        
        nearby_agents = [
            e for e in state.recent_events 
            if ":" not in e.subject and e.subject != state.name # Heuristic for agent vs object
        ]
        
        if not nearby_agents:
            return None
            
        target_event = nearby_agents[0] # Just pick first one for MVP
        target_agent_name = target_event.subject
        
        # Don't chat if already chatting
        if state.chatting_with or state.chatting_with_buffer.get(target_agent_name, 0) > 0:
            return None
            
        # 2. Decide to talk
        should_chat = self.talk_decider(
            context=f"{state.name} is at {state.current_tile}",
            current_time=str(state.time),
            init_agent=state.name,
            agent_with=target_agent_name,
            last_chat_summary="never chatted", # TODO from memory
            init_agent_observation=state.current_action.event.description if state.current_action else "idle",
            agent_with_observation=target_event.description
        )
        
        if should_chat:
            log_agent(state.name, f"Decided to chat with {target_agent_name}", "INFO")
            
            # Generate greeting
            utterance, end = self.dialogue_generator(
                agent=state.name,
                identity=state.identity_description,
                memory="Memory of agent...", # TODO
                past_context="Context...",
                location=str(state.current_tile),
                agent_action="Greeting",
                agent_with=target_agent_name,
                agent_with_action=target_event.description,
                conversation_history=""
            )
            
            # Create Chat Action
            s, p, o = self.event_parser.get_triple(state.name, f"chat with {target_agent_name}")
            
            chat_action = Action(
                address=f"<persona> {target_agent_name}",
                start_time=state.time.time,
                duration=10,
                emoji="💬",
                event=Event(subject=s, predicate=p, object_=o, description=f"chatting with {target_agent_name}", tile=state.current_tile, depth=0)
            )
            
            sig = ActionSignal(next_action=chat_action)
            sig.stop_chatting = end
            sig.update_chat_buffer = {target_agent_name: 60} # Cooldown
            
            return sig
            
        return None

    def _is_action_finished(self, state: AgentState) -> bool:
        """
        Checks if the current action is finished.
        Logic ported from Scratch.is_action_finished
        """
        if not state.current_action:
            return True

        if state.chatting_with:
             # Chatting logic is handled by specific chat duration in Scratch usually
             # For now, rely on end_task time
             pass

        start = state.current_action.start_time
        duration = state.current_action.duration
        
        # Safe handling if we don't have start time (should fail gracefully)
        if not start: return True
        
        if start.second != 0:
            start = start.replace(second=0) + datetime.timedelta(minutes=1)
        
        # Ensure duration is an int, as it might come from JSON as string
        try:
            duration_int = int(duration)
        except (ValueError, TypeError):
            duration_int = 60 # Default fallback
            
        end_time = start + datetime.timedelta(minutes=duration_int)
        
        if state.time.time >= end_time:
            return True
            
        return False

    def _determine_next_action_from_schedule(self, state: AgentState, schedule: List[Tuple[str, int]]) -> ActionSignal:
        """
        Logic from Plan._determine_action
        """
        curr_index = self._get_schedule_index(state, schedule)
        
        # TODO: Decomposition logic (splitting "sleeping" -> "sleeping", "reading")
        # For MVP, skipping complex decomposition and just taking the schedule item
        
        if curr_index >= len(schedule):
            # End of day / fall off
            action_desc = "sleeping"
            action_duration = 60
        else:
            action_desc, action_duration = schedule[curr_index]

        # Determine Location
        # Note: This logic requires SpatialMemory. 
        # Ideally SpatialMemory is passed in AgentState or we access it via a helper.
        # Since AgentState logic in neural_types.py didn't include full spatial memory tree,
        # we might need to assume we can access it or it's passed. 
        # For now, I'll use placeholders or assume the inputs are sufficient.
        
        # Simplified next action:
        # Generate triples
        subject, predicate, object_ = self.event_parser.get_triple(state.name, action_desc)
        
        next_action = Action(
            address="<random>", # Placeholder
            start_time=state.time.time,
            duration=action_duration,
            emoji=self.emoji_mapper(action_desc),
            event=Event(
                subject=subject,
                predicate=predicate,
                object_=object_,
                description=action_desc,
                tile=state.current_tile,
                depth=0
            )
        )
        
        return ActionSignal(next_action=next_action)

    def _get_schedule_index(self, state: AgentState, schedule: List[Tuple[str, int]]) -> int:
        # TODO: Implement accurate index calculation based on time elapsed
        # For now, returning 0 or calculated simple index
        return 0 
