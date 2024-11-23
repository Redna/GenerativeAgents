from dataclasses import asdict
import math
from operator import itemgetter

from langgraph.graph import StateGraph
from langgraph.constants import START, END, Send

from typing import Annotated, TypedDict

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.core.events import Event, EventType, PerceivedEvent
from generative_agents.simulation.maze import Level, Maze
from generative_agents.core.agent import Agent
from generative_agents.utils import logger

PERCEIVE_SPACE = "perceive_space"
PERCEIVE_EVENTS = "perceive_events"
STORE_EVENTS = "store_events"

def upsert(left: list, right: list):
    if left is None:
        left = []
    if right is None:
        right = []

    new_list = left

    for item in right:
        if item in new_list:
            new_list[new_list.index(item)] = item
        else:
            new_list += [item]

    return new_list

class PerceptionState(TypedDict):
    perceived_events: Annotated[list[PerceivedEvent], upsert]

class Perception:
    def __init__(self, agent: Agent, maze: Maze):
        self.agent = agent
        self.maze = maze

        workflow = StateGraph(PerceptionState)
        workflow.add_node(PERCEIVE_SPACE, self.perceive_space)
        workflow.add_node(PERCEIVE_EVENTS, self.perceive_events)
        workflow.add_node(STORE_EVENTS, self.store_events)
        workflow.add_edge(START, PERCEIVE_SPACE)
        workflow.add_edge(PERCEIVE_SPACE, PERCEIVE_EVENTS)
        workflow.add_conditional_edges(PERCEIVE_EVENTS, self.process_events, [STORE_EVENTS, END])
        workflow.add_edge(STORE_EVENTS, END)
        self.workflow = workflow

    def perceive_space(self, state: PerceptionState) -> PerceptionState:
        nearby_tiles = self.maze.get_nearby_tiles(self.agent.scratch.tile, self.agent.scratch.vision_radius)
        for tile in nearby_tiles:
            self.agent.spatial_memory.add(tile)

        logger.log(self.agent.name, f"Perceived {len(nearby_tiles)} tiles")
        return PerceptionState(perceived_events=[])

    def perceive_events(self, state: PerceptionState) -> PerceptionState:
        current_arena = self.agent.scratch.tile.get_path(Level.ARENA)
        percept_events_dict = dict()
        percept_events_list = []

        nearby_tiles = self.maze.get_nearby_tiles(self.agent.scratch.tile, self.agent.scratch.vision_radius)
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
                logger.log(e)
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))
        perceived_events = []
        for _, event in percept_events_list[:self.agent.scratch.attention_bandwith]:
            perceived_events += [event]

        logger.log(self.agent.name, f"Perceived {len(perceived_events)} events")
        return {"perceived_events": perceived_events}

    def process_events(self, state: PerceptionState) -> list:
        perceived_events = state.get("perceived_events", [])
        send_events = [Send(STORE_EVENTS, {"perceived_events": [perceived_event]}) for perceived_event in perceived_events] if perceived_events else [END]
        return send_events

    def store_events(self, state: PerceptionState):
        event = state["perceived_events"][0]

        if not event.predicate:
            event.predicate = "is"

        if not isinstance(event, PerceivedEvent) and event.event_type != EventType.CHAT:
            event.description = f"{event.subject.split(':')[-1]} {event.predicate} {event.object_}"
            event = self._perceive_event(event, type_=EventType.EVENT)

        if event.subject == self.agent.name and event.predicate == "chat with":
            event = self._perceive_event(event, type_=EventType.CHAT)
        self.agent.scratch.reflection_trigger_max -= event.poignancy * 10

        return {"perceived_events": [event]}

    def _perceive_event(self, event: Event, type_: EventType = EventType.EVENT):
        if not isinstance(event, PerceivedEvent):
            event_poignancy = self._rate_perception_poignancy(type_, event.description)

            logger.log(self.agent.name, f"'{event.description}' poignancy is {event_poignancy}")
            event_dict = asdict(event)
            event_dict.update({"event_type": type_})
            event = PerceivedEvent(**event_dict, poignancy=event_poignancy)
            event = self.agent.associative_memory.add(event)
        return event

    def _rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = rate_poignance(self.agent.name, self.agent.scratch.identity, event_type.value, description)
        return int(score) / 10