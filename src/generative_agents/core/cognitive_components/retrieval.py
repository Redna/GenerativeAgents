from functools import lru_cache
from typing import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END

from generative_agents.core.events import EventType, PerceivedEvent
from generative_agents.utils import logger

ADD_CURRENT_EVENT = "add_current_event"
RETRIEVE_EVENTS = "retrieve_events"
RETRIEVE_THOUGHTS = "retrieve_thoughts"

def upsert(left: dict, right: dict):
    if left is None:
        left = {}
    if right is None:
        right = {}

    left.update(right)
    return left


class RetrievalState(TypedDict):
    perceived_events: list[PerceivedEvent]
    retrieved: Annotated[dict[str, dict[str, list[PerceivedEvent]]], upsert]

class Retrieval:
    def __init__(self, agent):
        self.agent = agent
        workflow = StateGraph(RetrievalState)
        workflow.add_node(ADD_CURRENT_EVENT, self.add_current_event)
        workflow.add_node(RETRIEVE_EVENTS, self.retrieve_events)
        workflow.add_node(RETRIEVE_THOUGHTS, self.retrieve_thoughts)
        workflow.add_edge(START, ADD_CURRENT_EVENT)
        workflow.add_edge(ADD_CURRENT_EVENT, RETRIEVE_EVENTS)
        workflow.add_edge(ADD_CURRENT_EVENT, RETRIEVE_THOUGHTS)
        workflow.add_edge(RETRIEVE_EVENTS, END)
        workflow.add_edge(RETRIEVE_THOUGHTS, END)
        self.workflow = workflow

    def add_current_event(self, state: RetrievalState) -> RetrievalState:
        retrieved = state.get("retrieved", dict())
        for event in state.get("perceived_events", []):
            retrieved[event.description] = dict()
            retrieved[event.description]["curr_event"] = event

        logger.log(self.agent.name, f"Added {len(retrieved)} events to retrieval")
        return RetrievalState(retrieved=retrieved)

    def retrieve_events(self, state: RetrievalState) -> RetrievalState:
        retrieved = state.get("retrieved", dict())
        for event in state.get("perceived_events", []):
            retrieved[event.description]["events"] = self._get_related_events(event, EventType.EVENT)
        return RetrievalState(retrieved=retrieved)

    def retrieve_thoughts(self, state: RetrievalState) -> RetrievalState:
        retrieved = state.get("retrieved", dict())
        for event in state.get("perceived_events", []):
            retrieved[event.description]["thoughts"] = self._get_related_events(event, EventType.THOUGHT)
        return RetrievalState(retrieved=retrieved)

    @lru_cache(maxsize=2048)
    def _get_related_to_text(self, text: str,  event_type: EventType = None):
        if event_type:
            memories = self.agent.associative_memory.retrieve_relevant_entries_by_type(text, event_type)
        else:
            memories = self.agent.associative_memory.retrieve_relevant_entries(text)

        return memories

    def _get_related_events(self, event: PerceivedEvent, event_type: EventType = None):
        return self._get_related_to_text(event.description, event_type)