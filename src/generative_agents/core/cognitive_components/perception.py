


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
class Perception:
    def __init__(self, agent):
        self.agent = agent

    @component.output_types(perceived_events=list[PerceivedEvent])
    def run(self, maze: Maze) -> list[PerceivedEvent]:
        """
        Perceives events around the persona and saves it to the memory, both events 
        and spaces. 

        We first perceive the events nearby the persona, as determined by its 
        <vision_radius>. If there are a lot of events happening within that radius, we 
        take the <att_bandwidth> of the closest events. Finally, we check whether
        any of them are new, as determined by <retention>. If they are new, then we
        save those and return the <ConceptNode> instances for those events. 

        INPUT: 
            maze: An instance of <Maze> that represents the current maze in which the 
                persona is acting in. 
        OUTPUT: 
            ret_events: a list of <ConceptNode> that are perceived and new. 
        """
        # PERCEIVE SPACE
        # We get the nearby tiles given our current tile and the persona's vision
        # radius.
        nearby_tiles = maze.get_nearby_tiles(self.scratch.tile,
                                             self.scratch.vision_radius)

        # Perceive all nearby tiles and store it in the spatial memory if not already done
        for tile in nearby_tiles:
            self.spatial_memory.add(tile)

        # PERCEIVE EVENTS.
        # We will perceive events that take place in the same arena as the
        # persona's current arena.
        current_arena = self.scratch.tile.get_path(Level.ARENA)

        # We do not perceive the same event twice (this can happen if an object_ is
        # extended across multiple tiles).
        percept_events_dict = dict()
        # We will order our percept based on the distance, with the closest ones
        # getting priorities.
        percept_events_list = []
        # First, we put all events that are occuring in the nearby tiles into the
        # percept_events_list
        for tile in nearby_tiles:
            if not tile.events or tile.get_path(Level.ARENA) != current_arena:
                continue

            # This calculates the distance between the persona's current tile,
            # and the target tile.
            dist = math.dist([tile.x, tile.y], [
                             self.scratch.tile.x, self.scratch.tile.y])

            # Add any relevant events to our temp set/list with the distant info.
            
            try:
                for event in tile.events.values():
                    if event.spo_summary not in percept_events_dict:
                        percept_events_list += [[dist, event]]
                        percept_events_dict[event.spo_summary] = event
                        whisper(self.name, f"nearby event {event.description}")
            except Exception as e:
                print(e)
        # We sort, and perceive only persona.scratch.att_bandwidth of the closest
        # events. If the bandwidth is larger, then it means the persona can perceive
        # more elements within a small area.
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))
        perceived_events = []
        for dist, event in percept_events_list[:self.scratch.attention_bandwith]:
            perceived_events += [event]

        # Storing events.
        # <ret_events> is a list of <ConceptNode> instances from the persona's
        # associative memory.
        final_events = []
        perception_tasks = []
        
        for perceived_event in perceived_events:
            event = deepcopy(perceived_event)

            if not event.predicate:
                # If the object_ is not present, then we default the event to "idle".
                event.predicate = "is"

            whisper(self.name, f"{event.description}")

            if type(event) != PerceivedEvent or event.event_type != EventType.CHAT:
                event.description = f"{event.subject.split(':')[-1]} is {event.description}"

            # We retrieve the latest persona.scratch.retention events. If there is
            # something new that is happening (that is, p_event not in latest_events),
            # then we add that event to the a_mem and return it.

            latest_events = self.associative_memory.latest_events_summary

            if event.spo_summary not in latest_events:
                # We start by managing keywords.
                if ":" in event.object_:
                    event.object_ = event.object_.split(":")[-1]

            if "(" in event.description:
                event.description = (event.description.split("(")[1]
                                     .split(")")[0]
                                     .strip())

            # If we observe the persona's self chat, we include that in the memory
            # of the persona here.
            final_events = []

            if event.subject == self.name and event.predicate == "chat with":
                final_events += [self._perceive_event(event, type_=EventType.CHAT)]
            else:
                final_events += [event]

        for event in final_events:
            self.scratch.reflection_trigger_max -= event.poignancy * 10

        return final_events
    
    def _perceive_event(self, event: Event, type_: EventType = EventType.EVENT):
        if type(event) != PerceivedEvent:
            event_poignancy = self._rate_perception_poignancy(type_, event.description)

            whisper(self.name, f"event poignancy is {event_poignancy}")
            event = PerceivedEvent(**asdict(event), event_type=type_, poignancy=event_poignancy)
            event = self.associative_memory.add(event)
        return event
    
    @lru_cache(maxsize=512)
    async def _rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1
        
        score = rate_poignance(self.name, await self.scratch.identity, event_type.value, description)
        return int(score) / 10