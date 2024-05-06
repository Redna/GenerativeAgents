


from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import math
from operator import itemgetter
from haystack import Pipeline, component

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.core.events import Event, EventType, PerceivedEvent
from generative_agents.core.whisper.whisper import whisper
from generative_agents.simulation.maze import Level, Maze


@component
class Retrieval:
    def __init__(self, agent):
        self.agent = agent

    @component.output_types(perceived_events=dict[str, dict[str, list[PerceivedEvent]]])
    def run(self, perceived: list[PerceivedEvent]) -> dict[str, dict[str, list[PerceivedEvent]]]:
        retrieved = dict()

        for event in perceived:
            retrieved[event.description] = dict()
            retrieved[event.description]["curr_event"] = event
            retrieved[event.description]["events"] = self._get_related_events(
                event, EventType.EVENT)
            retrieved[event.description]["thoughts"] = self._get_related_events(
                event, EventType.THOUGHT)

            whisper(
                self.name, f"{event.description} has {len(retrieved[event.description]['events'])} related events and {len(retrieved[event.description]['thoughts'])} related thoughts")
        return retrieved