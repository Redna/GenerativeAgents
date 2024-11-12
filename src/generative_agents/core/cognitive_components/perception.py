from dataclasses import asdict
import math
from operator import itemgetter

from langgraph.graph import StateGraph
from langgraph.constants import START, END, Send

from typing import TypedDict

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.core.events import Event, EventType, PerceivedEvent
from generative_agents.core.whisper.whisper import whisper
from generative_agents.simulation.maze import Level, Maze
from generative_agents.core.agent import Agent

class PerceptionState(TypedDict):
    perceived_events: list[PerceivedEvent]

class Perception:
    def __init__(self, agent: Agent):
        self.agent = agent

        workflow = StateGraph(PerceptionState)
        workflow.add_node("perceive_space", self.perceive_space)
        workflow.add_node("perceive_events", self.perceive_events)
        workflow.add_node("store_events", self.store_events)
        workflow.add_edge(START, "perceive_space")
        workflow.add_edge("perceive_space", "perceive_events")
        workflow.add_conditional_edges("process_events", self.process_events, ["store_events"])
        workflow.add_edge("store_events", END)
        self.workflow = workflow

    def perceive_space(self, maze: Maze):
        nearby_tiles = maze.get_nearby_tiles(self.agent.scratch.tile, self.agent.scratch.vision_radius)
        for tile in nearby_tiles:
            self.agent.spatial_memory.add(tile)

    def perceive_events(self, maze: Maze):
        current_arena = self.agent.scratch.tile.get_path(Level.ARENA)
        percept_events_dict = dict()
        percept_events_list = []
        nearby_tiles = maze.get_nearby_tiles(self.agent.scratch.tile, self.agent.scratch.vision_radius)
        for tile in nearby_tiles:
            if not tile.events or tile.get_path(Level.ARENA) != current_arena:
                continue
            dist = math.dist([tile.x, tile.y], [self.agent.scratch.tile.x, self.agent.scratch.tile.y])
            try:
                for event in tile.events.values():
                    if event.spo_summary not in percept_events_dict:
                        percept_events_list += [[dist, event]]
                        percept_events_dict[event.spo_summary] = event
            except Exception as e:
                print(e)
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))
        perceived_events = []
        for dist, event in percept_events_list[:self.agent.scratch.attention_bandwith]:
            perceived_events += [event]
        return perceived_events

    def process_events(self, state: PerceptionState):
        perceived_events = state["perceived_events"]
        return [Send("store_events", {"perceived_events": [perceived_event]}) for perceived_event in perceived_events]

    def store_events(self, state: PerceptionState):
        event = state["perceived_events"][0]

        if not event.predicate:
            event.predicate = "is"

        if not isinstance(event, PerceivedEvent) or event.event_type != EventType.CHAT:
            event = self._perceive_event(event, type_=EventType.EVENT)
            event.description = f"{event.subject.split(':')[-1]} is {event.description}"


        if event.subject == self.agent.name and event.predicate == "chat with":
            event = self._perceive_event(event, type_=EventType.CHAT)
        self.agent.scratch.reflection_trigger_max -= event.poignancy * 10

        return {"perceived_events": [event]}

    def _perceive_event(self, event: Event, type_: EventType = EventType.EVENT):
        if type(event) != PerceivedEvent:
            event_poignancy = self._rate_perception_poignancy(type_, event.description)

            whisper(self.agent.name, f"event poignancy is {event_poignancy}")
            event = PerceivedEvent(**asdict(event), event_type=type_, poignancy=event_poignancy)
            event = self.agent.associative_memory.add(event)
        return event

    def _rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = rate_poignance(self.agent.name, self.agent.scratch.identity, event_type.value, description)
        return int(score) / 10