


from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import math
from operator import itemgetter
from haystack import Pipeline, component
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.core.events import Event, EventType, PerceivedEvent
from generative_agents.core.whisper.whisper import whisper
from generative_agents.simulation.maze import Level, Maze
from generative_agents.utils import timeit

from generative_agents.persistence import database

class RetrievalState(TypedDict):
    retrieved: dict[str, dict[str, list[PerceivedEvent]]]

@component
class Retrieval:
    def __init__(self, agent):
        self.agent = agent
        workflow = StateGraph(RetrievalState)
        workflow.add_node(self.add_current_event)
        workflow.add_node(self.retrieve_events)
        workflow.add_node(self.retrieve_thoughts)
        workflow.add_edge(START, self.add_current_event)
        workflow.add_edge(self.add_current_event, self.retrieve_events)
        workflow.add_edge(self.retrieve_events, self.retrieve_thoughts)
        workflow.add_edge(self.retrieve_thoughts, END)
        self.workflow = workflow

    def add_current_event(self, state: RetrievalState) -> RetrievalState:
        for event in state["perceived"]:
            state["retrieved"][event.description] = dict()
            state["retrieved"][event.description]["curr_event"] = event
        return state

    def retrieve_events(self, state: RetrievalState) -> RetrievalState:
        for event in state["perceived"]:
            state["retrieved"][event.description]["events"] = self._get_related_events(event, EventType.EVENT)
        return state

    def retrieve_thoughts(self, state: RetrievalState) -> RetrievalState:
        for event in state["perceived"]:
            state["retrieved"][event.description]["thoughts"] = self._get_related_events(event, EventType.THOUGHT)
        return state

    def _get_related_to_text(self, text: str,  event_type: EventType = None):
        if event_type:
            memories = database.get_by_type(self.agent.name, text, event_type)
        else:
            memories = database.get(self.agent.name, text)

        return [PerceivedEvent.from_db_entry(memory) for memory in memories]

    def _get_related_events(self, event: PerceivedEvent, event_type: EventType = None):
        return self._get_related_to_text(event.description, event_type)