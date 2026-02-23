import asyncio
from generative_agents.common.events import PerceivedEvent, EventType
from generative_agents.intelligence.modules.dialogue import (
    DialogueGenerator, DialogueSummarizer, DialoguePlanner
)
from generative_agents.common.logging import log_agent

class DialogueCoordinator:
    def __init__(self):
        self.generator = DialogueGenerator()
        self.summarizer = DialogueSummarizer()
        self.planner = DialoguePlanner()
        self.max_turns = 6

    async def run_conversation_async(self, initiator, target, opening_line: str):
        """Runs the conversation in the background without blocking the simulation tick."""
        log_agent("System", f"Background chat started: {initiator.name} & {target.name}", "INFO")
        
        history = [f"{initiator.name}: {opening_line}"]
        active_speaker, listener = target, initiator
        
        for turn in range(self.max_turns):
            memory_context = f"{active_speaker.name} is chatting with {listener.name}."
            past_context = "No recent interactions."
            
            # Offload synchronous DSPy call to a thread so asyncio loop keeps spinning
            utterance_result = await asyncio.to_thread(
                self.generator,
                active_speaker.name,
                active_speaker.working_memory.identity_description,
                memory_context,
                past_context,
                active_speaker.working_memory.tile.get_unique_name() if active_speaker.working_memory.tile else "", # location
                active_speaker.activity if hasattr(active_speaker, 'activity') else "",
                listener.name,
                listener.activity if hasattr(listener, 'activity') else "",
                "\n".join(history)
            )
            
            # Assuming DialogueGenerator returns a tuple of (utterance, end_convo)
            # Adjust if the signature returns something different based on the actual module
            utterance = utterance_result[0] if isinstance(utterance_result, tuple) else utterance_result
            end_convo = utterance_result[1] if isinstance(utterance_result, tuple) and len(utterance_result) > 1 else False
            
            if hasattr(utterance_result, 'utterance'):
                utterance = utterance_result.utterance
            if hasattr(utterance_result, 'end_convo'):
                end_convo = utterance_result.end_convo
            
            history.append(f"{active_speaker.name}: {utterance}")
            
            # ---> NEW: Broadcast the utterance to the physical map <---
            if getattr(active_speaker.working_memory, 'action', None) and active_speaker.working_memory.action.event:
                # Format exactly like this so the perception layer can parse it later
                active_speaker.working_memory.action.event.description = f"chatting with {listener.name}, saying: '{utterance}'"
            
            from generative_agents.common import global_state
            # Add a slight artificial delay so the text lingers on the map for at least one simulation tick
            if hasattr(global_state, 'time') and getattr(global_state.time, 'increment', None):
                await asyncio.sleep(global_state.time.increment)
            else:
                 await asyncio.sleep(1.0) # Fallback

            if end_convo or turn == self.max_turns - 1:
                break
                
            active_speaker, listener = listener, active_speaker
            
        await self._process_aftermath_async(initiator, target, history)

    async def _process_aftermath_async(self, agent_a, agent_b, history: list[str]):
        """Summarizes the chat, injects memories, and cleanly releases the agents."""
        full_chat = "\n".join(history)
        
        # Offload summary and planning to threads
        summary = await asyncio.to_thread(self.summarizer, full_chat)
        takeaway_a = await asyncio.to_thread(self.planner, agent_a.name, full_chat)
        takeaway_b = await asyncio.to_thread(self.planner, agent_b.name, full_chat)
        
        # Inject memories directly into their repositories
        self._inject_memory(agent_a, agent_b, summary if isinstance(summary, str) else str(summary), takeaway_a if isinstance(takeaway_a, str) else str(takeaway_a))
        self._inject_memory(agent_b, agent_a, summary if isinstance(summary, str) else str(summary), takeaway_b if isinstance(takeaway_b, str) else str(takeaway_b))
        
        # Cleanly release the agents from their chatting state
        agent_a.working_memory.chatting_with = ""
        agent_b.working_memory.chatting_with = ""
        agent_a.working_memory.action = None
        agent_b.working_memory.action = None
        
        log_agent("System", f"Background chat resolved. Summary: {summary}", "INFO")

    def _inject_memory(self, agent, other_agent, summary: str, takeaway: str):
        """Creates formal memory events from the conversation."""
        summary_event = PerceivedEvent(
            event_type=EventType.CHAT,
            poignancy=0.8,
            depth=1,
            description=f"Had a conversation with {other_agent.name}. Summary: {summary}",
            entity_id=agent.name,
        )
        agent.memory.add(summary_event)
        
        takeaway_event = PerceivedEvent(
            event_type=EventType.THOUGHT,
            poignancy=0.7,
            depth=2,
            description=f"After talking to {other_agent.name}, I need to remember: {takeaway}",
            entity_id=agent.name,
        )
        agent.memory.add(takeaway_event)
