import datetime
import math
from operator import itemgetter
import random
from typing import Dict, List, Tuple
from communication.models import AgentDTO, MovementDTO
from memory import AssociativeMemory, MemoryTree, Scratch
from simulation.maze import Level, Maze, Tile
from conversational.chain import (poignance_chain, wake_up_hour_chain, 
                                  first_daily_plan_chain, 
                                  daily_plan_and_status_chain, 
                                  action_sector_locations_chain,
                                  task_decomposition_chain,
                                  action_arena_locations_chain,
                                  action_location_game_object_chain,
                                  action_pronunciatio_chain,
                                  action_event_triple_chain,
                                  object_event_chain, 
                                  action_object_pronunciatio_chain)
from conversational.chain.hourly_breakdown import HourlyBreakdown
from persistence import database
from simulation.time import SimulationTime, DayType
from core.events import Action, Event, ObjectAction, EventType, PerceivedEvent

class Agent():
    def __init__(self, name: str, description: str, location: str, emoji: str, activity: str, tile: Tile):
        self.name = name
        self.description = description
        self.location = location
        self.emoji = emoji
        self.activity = activity
        self.scratch = Scratch()
        self.spatial_memory = MemoryTree()
        self.associative_memory = AssociativeMemory(self.name, self.scratch.retention)
        self.scratch.tile = tile

    def to_dto(self):
        return AgentDTO(
            name=self.name,
            description=self.description,
            location=self.location,
            emoji=self.emoji,
            activity=self.activity,
            movement=MovementDTO(col=self.scratch.tile.x,
                                 row=self.scratch.tile.y)
        )

    @staticmethod
    def from_dto(dto: AgentDTO, maze: Maze):
        return Agent(name=dto.name,
                     description=dto.description,
                     location=dto.location,
                     emoji=dto.emoji,
                     activity=dto.activity,
                     tile=maze.get_tile(dto.movement.col, dto.movement.row))

    def update(self, time: SimulationTime, maze: Maze, agents: Dict[str, 'Agent']):
        # random movement
        # if not self.scratch.planned_path:
        #    self.scratch.planned_path = self.__get_random_path(maze)
        #
        # self.agent.tile = self.scratch.planned_path.pop(0)
        # print("Moving to", self.tile)

        daytype: DayType = DayType.SAME_DAY

        if not self.scratch.time:
            daytype = daytype.FIRST_DAY
        elif (self.scratch.time.today != time.today):
            daytype = daytype.NEW_DAY

        self.scratch.time = time

        perceived = self.perceive(maze)
        retrieved = self.retrieve(perceived)
        plan = self.plan(maze, agents, daytype, retrieved)
        self.reflect()

        return self.execute(maze, agents, plan)

    def perceive(self, maze: Maze):
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
        for nearby_tile in nearby_tiles:
            self.spatial_memory.add(nearby_tile)

        # PERCEIVE EVENTS.
        # We will perceive events that take place in the same arena as the
        # persona's current arena.
        current_arena = self.scratch.tile.get_path(Level.ARENA)

        # We do not perceive the same event twice (this can happen if an object_ is
        # extended across multiple tiles).
        percept_events_set = set()
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
            dist = math.dist([tile.x, tile.y], [self.tile.x, self.tile.y])

            # Add any relevant events to our temp set/list with the distant info.
            for event in tile.events:
                if event not in percept_events_set:
                    percept_events_list += [[dist, event]]
                    percept_events_set.add(event)

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

        # "subject": "the Ville:Hobbs Cafe:cafe:kitchen sink", "predicate": "is", "object_": "idle", "description": "kitchen sink is idle"

        for perceived_event in perceived_events:
            event = perceived_event.copy()

            if not event.predicate:
                # If the object_ is not present, then we default the event to "idle".
                event.predicate = "is"
                event.object_ = "idle"

            event.description = f"{event.subject.split(':')[-1]} is {event.description}"

            # We retrieve the latest persona.scratch.retention events. If there is
            # something new that is happening (that is, p_event not in latest_events),
            # then we add that event to the a_mem and return it.

            latest_events = self.associative_memory.latest_events_summary

            if event.spo_summary not in latest_events:
                # We start by managing keywords.
                if ":" in event.subject:
                    event.subject = event.subject.split(":")[-1]
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
                active_event = self.scratch.action.event
                poignancy = self.rate_perception_poignancy(
                    EventType.CHAT, active_event.description)
                perceived_event = PerceivedEvent(
                    **active_event.dict(), event_type=EventType.CHAT, poignancy=poignancy)
                final_events += [self.associative_memory.add(perceived_event)]

            # We rate the poignancy of the event.
            event_poignancy = self.rate_perception_poignancy(
                EventType.EVENT, event.description)
            perceived_event = PerceivedEvent(
                **event.dict(), event_type=EventType.EVENT, poignancy=event_poignancy)
            # Finally, we add the current event to the agent's memory.
            final_events += [self.associative_memory.add(perceived_event)]

        return final_events

    def rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = poignance_chain.run(agent_name=self.name,
                                    agent_description=self.description,
                                    description=description,
                                    event_type=event_type.value)

        # TODO properly check the output here
        return int(score) / 10

    def retrieve(self, perceived: List[PerceivedEvent]) -> Dict[str, Dict[str, List[PerceivedEvent]]]:
        retrieved = dict()

        for event in perceived:
            retrieved[event.description] = dict()
            retrieved[event.description]["curr_event"] = [event]
            retrieved[event.description]["events"] = self._get_related_events(
                event, EventType.EVENT)
            retrieved[event.description]["thoughts"] = self._get_related_events(
                event, EventType.THOUGHT)

        return retrieved

    def plan(self, maze: Maze, agents: Dict[str, 'Agent'], daytype: DayType, retrieved: Dict[str, Dict[str, List[PerceivedEvent]]]):
        if daytype == DayType.NEW_DAY:
            self._long_term_planning(daytype)

        if self.scratch.is_action_finished():
            self._determine_action(maze)

         # PART 3: If you perceived an event that needs to be responded to (saw 
        # another persona), and retrieved relevant information. 
        # Step 1: Retrieved may have multiple events represented in it. The first 
        #         job here is to determine which of the events we want to focus 
        #         on for the persona. 
        #         <focused_event> takes the form of a dictionary like this: 
        #         dictionary {["curr_event"] = <ConceptNode>, 
        #                     ["events"] = [<ConceptNode>, ...], 
        #                     ["thoughts"] = [<ConceptNode>, ...]}

        focused_event = False
        if retrieved.keys(): 
            focused_event = self._choose_retrieved(retrieved)
        
        # Step 2: Once we choose an event, we need to determine whether the
        #         persona will take any actions for the perceived event. There are
        #         three possible modes of reaction returned by _should_react. 
        #         a) "chat with {target_persona.name}"
        #         b) "react"
        #         c) False
        if focused_event: 
            reaction_mode = self._should_react(focused_event, agents)
            if reaction_mode: 
                # If we do want to chat, then we generate conversation 
                if reaction_mode[:9] == "chat with":
                    _chat_react(maze, persona, focused_event, reaction_mode, personas)
                elif reaction_mode[:4] == "wait": 
                    _wait_react(persona, reaction_mode)
                # elif reaction_mode == "do other things": 
                #   _chat_react(persona, focused_event, reaction_mode, personas)

        # Step 3: Chat-related state clean up. 
        # If the persona is not chatting with anyone, we clean up any of the 
        # chat-related states here. 
        if persona.scratch.act_event[1] != "chat with":
            persona.scratch.chatting_with = None
            persona.scratch.chat = None
            persona.scratch.chatting_end_time = None
        # We want to make sure that the persona does not keep conversing with each
        # other in an infinite loop. So, chatting_with_buffer maintains a form of 
        # buffer that makes the persona wait from talking to the same target 
        # immediately after chatting once. We keep track of the buffer value here. 
        curr_persona_chat_buffer = persona.scratch.chatting_with_buffer
        for persona_name, buffer_count in curr_persona_chat_buffer.items():
            if persona_name != persona.scratch.chatting_with: 
            persona.scratch.chatting_with_buffer[persona_name] -= 1

        return persona.scratch.act_address

    def reflect(self):
        pass

    def execute(self, maze: Maze, agents: Dict[str, 'Agent'], plan):
        pass

    def _should_react(self, focused_event: Dict[str, List[PerceivedEvent]], agents: Dict[str, 'Agent']):
        """
        Determines what form of reaction the persona should exihibit given the 
        retrieved values. 
        INPUT
            persona: Current <Persona> instance whose action we are determining. 
            retrieved: A dictionary of <ConceptNode> that were retrieved from the 
                    the persona's associative memory. This dictionary takes the
                    following form: 
                    dictionary[event.description] = 
                        {["curr_event"] = <ConceptNode>, 
                        ["events"] = [<ConceptNode>, ...], 
                        ["thoughts"] = [<ConceptNode>, ...] }
            personas: A dictionary that contains all persona names as keys, and the 
                    <Persona> instance as values. 
        """
        def lets_talk(init_agent: Agent, target_agent: Agent, retrieved: Dict[str, List[PerceivedEvent]]):
            if (not target_agent.scratch.action.address 
                or not target_agent.scratch.action.event
                or not init_agent.scratch.action.address
                or not init_agent.scratch.action.event): 
                return False

            if ("sleeping" in target_agent.scratch.action.event.description 
                or "sleeping" in init_agent.scratch.action.event.description): 
                return False

            if init_agent.scratch.time.hour == 23: 
                return False

            if "<waiting>" in target_agent.scratch.action.address:
                return False

            if (target_agent.scratch.chatting_with or init_agent.scratch.chatting_with): 
                return False

            if (target_agent.name in init_agent.scratch.chatting_with_buffer): 
                if init_agent.scratch.chatting_with_buffer[target_agent.name] > 0: 
                    return False

            if self._generate_decide_to_talk(init_agent, target_agent, retrieved): 
                return True

            return False

        def lets_react(init_agent, target_agent, retrieved): 
            if (not target_agent.scratch.act_address 
                or not target_agent.scratch.act_description
                or not init_agent.scratch.act_address
                or not init_agent.scratch.act_description): 
                return False

            if ("sleeping" in target_agent.scratch.act_description 
                or "sleeping" in init_agent.scratch.act_description): 
                return False

            # return False
            if init_agent.scratch.curr_time.hour == 23: 
                return False

            if "waiting" in target_agent.scratch.act_description: 
                return False
            if init_agent.scratch.planned_path == []:
                return False

            if (init_agent.scratch.act_address 
                != target_agent.scratch.act_address): 
                return False

            react_mode = self._generate_decide_to_react(init_agent, 
                                                target_agent, retrieved)

            if react_mode == "1": 
                wait_until = ((target_agent.scratch.act_start_time 
                    + datetime.timedelta(minutes=target_agent.scratch.act_duration - 1))
                    .strftime("%B %d, %Y, %H:%M:%S"))
                return f"wait: {wait_until}"
            elif react_mode == "2":
                return False
                return "do other things"
            else:
                return False #"keep" 

        # If the persona is chatting right now, default to no reaction 
        if self.scratch.chatting_with: 
            return False
        if "<waiting>" in self.scratch.act_address: 
            return False

        # Recall that retrieved takes the following form: 
        # dictionary {["curr_event"] = <ConceptNode>, 
        #             ["events"] = [<ConceptNode>, ...], 
        #             ["thoughts"] = [<ConceptNode>, ...]}
        curr_event = focused_event["curr_event"]

        if ":" not in curr_event.subject: 
            # this is a persona event. 
            if lets_talk(self, agents[curr_event.subject], focused_event):
                return f"chat with {curr_event.subject}"
            
            return lets_react(self, agents[curr_event.subject], focused_event)

        return False
    
    @staticmethod
    def _generate_decide_to_talk(init_agent: 'Agent', target_agent: 'Agent', retrieved: Dict[str, List[PerceivedEvent]]):
        
        pass

    @staticmethod
    def _generate_decide_to_react(init_agent: 'Agent', target_agent: 'Agent', retrieved: Dict[str, List[PerceivedEvent]]):
        pass
    
    def _choose_retrieved(self, retrieved: Dict[str, Dict[str, List[PerceivedEvent]]]):
        """
        Retrieved elements have multiple core "curr_events". We need to choose one
        event to which we are going to react to. We pick that event here. 
        INPUT
            persona: Current <Persona> instance whose action we are determining. 
            retrieved: A dictionary of <ConceptNode> that were retrieved from the 
                    the persona's associative memory. This dictionary takes the
                    following form: 
                    dictionary[event.description] = 
                        {["curr_event"] = <ConceptNode>, 
                        ["events"] = [<ConceptNode>, ...], 
                        ["thoughts"] = [<ConceptNode>, ...] }
        """
        no_self_event_retrieved = {description: context for description, context in retrieved.items() if context["curr_event"].subject != self.name}

        persona_context = [context for _, context in no_self_event_retrieved.items() if ":" not in context["curr_event"].subject]
        if persona_context:
            return random.choice(persona_context)

        non_idle_context = [context for _, context in no_self_event_retrieved.items() if "idle" not in context["curr_event"].description]
        if non_idle_context:
            return random.choice(non_idle_context)
        
        return None

    def _determine_action(self, maze):

        def needs_decomposition(action_description: str, action_duration: int):
            # TODO reformulate this logic

            if "sleep" not in action_description and "bed" not in action_description:
                return True
            elif "sleeping" in action_description or "asleep" in action_description or "in bed" in action_description:
                return False
            elif "sleep" in action_description or "bed" in action_description:
                if action_duration > 60:
                    return False
                return True

        current_index = self.scratch.get_daily_schedule_index()
        next_hour_index = self.scratch.get_daily_schedule_index(advance=60)

        # * Decompose *
        # During the first hour of the day, we need to decompose two hours
        # sequence. We do that here.
        if current_index == 0:
            # This portion is invoked if it is the first hour of the day.
            action_description, action_duration = self.scratch.daily_schedule[current_index]

            if action_duration >= 60:
                # We decompose if the next action is longer than an hour, and fits the
                # criteria described in determine_decomp.
                if needs_decomposition(action_description, action_duration):
                    self.scratch.daily_schedule[current_index:current_index +
                                                1] = self._decompose_action(current_index, action_description, action_duration)

            if next_hour_index + 1 < len(self.scratch.daily_schedule):
                action_description, action_duration = self.scratch.daily_schedule[next_hour_index]

                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        self.scratch.daily_schedule[next_hour_index:next_hour_index + 1] = self._decompose_action(next_hour_index, action_description, action_duration)

        if next_hour_index < len(self.scratch.daily_schedule):
            # If it is not the first hour of the day, this is always invoked (it is
            # also invoked during the first hour of the day -- to double up so we can
            # decompose two hours in one go). Of course, we need to have something to
            # decompose as well, so we check for that too.
            if self.scratch.time.hour < 23:
                # And we don't want to decompose after 11 pm.
                action_description, action_duration = self.scratch.daily_schedule[next_hour_index]
                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        self.scratch.daily_schedule[next_hour_index:next_hour_index+1] = (
                            self._decompose_action(next_hour_index, action_description, action_description))

        # * End of Decompose *

        # TODO find out what this is for?!
        # 1440
        # x_emergency = 0
        # for i in persona.scratch.f_daily_schedule:
        #    x_emergency += i[1]
        # print ("x_emergency", x_emergency)

        # if 1440 - x_emergency > 0:
        #    print ("x_emergency__AAA", x_emergency)
        # persona.scratch.f_daily_schedule += [["sleeping", 1440 - x_emergency]]

        action_description, action_duration = self.scratch.daily_schedule[current_index]
        
        action_sector = self._generate_next_action_sector(action_description)
        action_arena = self._generate_next_action_arena(action_description, action_sector)
        action_game_object = self._generate_next_action_game_object(action_description, action_arena)
        # new_address = action_game_object
        
        action_pronouncio = self._generate_action_pronunciatio(action_description)
        action_event = self._generate_action_event_triple(action_description)
        
        action_object_desctiption = self._generate_action_object_description(action_game_object, action_description)
        action_object_pronunciatio = self._generate_action_pronunciatio(action_object_desctiption)
        subject, predicate, object_ = self._generate_action_object_event_triple(action_game_object, action_object_desctiption)

        object_action = ObjectAction(address=action_game_object, 
                                    emoji=action_object_pronunciatio,
                                    event=Event(subject=subject,
                                                predicate=predicate,
                                                object_=object_,
                                                description=action_object_desctiption))
        
        next_action = Action(address=action_game_object,
                             start_time=self.scratch.time,
                             duration=action_duration,
                             emoji=action_pronouncio,
                             event=Event(subject=self.name,
                                         predicate=action_event[1],
                                         object_=action_event[2],
                                         description=action_description
                                         ),
                             object_action=object_action)
        
        self.scratch.action_queue.put(next_action)

    def _generate_action_object_event_triple(self, action_game_object, action_object_description):
        action_object_event_triple = action_event_triple_chain.run(agent_name=action_game_object,
                                                                   action_description=action_object_description)
        # TODO parse this triple
        return action_object_event_triple

    def _generate_action_object_description(self, action_game_object, action_description):
        action_object_description = object_event_chain.run(action_game_object=action_game_object,
                                                                        action_description=action_description)
        # TODO parse this output
        return action_object_description

    def _generate_action_event_triple(self, action_description):
        action_event = action_event_triple_chain.run(agent_name=self.name,
                                                     action_description=action_description)

        # TODO parse this triple
        return action_event

    def _generate_action_pronunciatio(self, action_description):
        action_pronouncio = action_pronunciatio_chain.run(action_description=action_description)
        return action_pronouncio
    
    def _generate_next_action_game_object(self, action_description, action_arena):
        available_objects = self.spatial_memory.get_str_accessible_arena_game_object_s(action_arena)
        
        game_object = action_location_game_object_chain.run(action_description=action_description, available_objects=available_objects)
        return f"{action_arena}:{game_object}"

    def _generate_next_action_arena(self, action_description, action_sector):
        name = self.name
        current_sector = self.scratch.tile.sector
        current_area = self.scratch.tile.arena
        sector = action_sector.split(":")[-1]
        
        arena = action_arena_locations_chain.run(agent_name=name,
                                                current_area=current_area,
                                                current_sector=current_sector,
                                                sector=sector,
                                                sector_arenas=self.spatial_memory.get_str_accessible_sector_arenas(action_sector),
                                                action_description=action_description) 
        
        return f"{action_sector}:{arena}"
        
    def _generate_next_action_sector(self, action_description):
        name = self.name
        home = self.scratch.home.sector
        home_arenas = self.spatial_memory.get_str_accessible_sector_arenas(home.get_path(Level.SECTOR))
        current_sector = self.scratch.tile.sector
        current_sector_arenas = self.spatial_memory.get_str_accessible_sector_arenas(self.scratch.tile.get_path(Level.SECTOR))
        nearby_sectors = self.spatial_memory.get_str_accessible_sectors(self.scratch.tile.get_path(Level.WORLD))
                
        
        next_sector = action_sector_locations_chain.run(agent_name=name,
                                                agent_home=home,
                                                agent_home_arenas=home_arenas,
                                                agent_current_sector=current_sector,
                                                agent_current_sector_arenas=current_sector_arenas,
                                                available_sectors_nearby=nearby_sectors,
                                                curr_action_description=action_description)
        
        return f"{self.scratch.tile.world}:{next_sector}"

    
    def _decompose_action(self, action_index: int, action_description: str, action_duration: int):
        """
        A few shot decomposition of a task given the task description 

        Persona state: identity stable set, curr_date_str, first_name

        The task is decomposed based on its relative position between the next tasks
        and the previous tasks.

        INPUT: 
            persona: The Persona class instance 
            task: the description of the task at hand in str form
                (e.g., "waking up and starting her morning routine")
            duration: an integer that indicates the number of minutes this task is 
                    meant to last (e.g., 60)
        OUTPUT: 
            a list of list where the inner list contains the decomposed task 
            description and the number of minutes the task is supposed to last. 
        EXAMPLE OUTPUT: 
            [['going to the bathroom', 5], ['getting dressed', 5], 
            ['eating breakfast', 15], ['checking her email', 5], 
            ['getting her supplies ready for the day', 15], 
            ['starting to work on her painting', 15]] 
          """
        current_index = self.scratch.get_daily_schedule_index()

        task_ids = []
        # if curr_f_org_index > 0:
        #   all_indices += [curr_f_org_index-1]
        task_ids += [current_index]
        if current_index+1 <= len(self.scratch.daily_schedule):
            task_ids += [current_index+1]
        if current_index+2 <= len(self.scratch.daily_schedule):
            task_ids += [current_index+2]

        task_sumaries = []
        for task_id in task_ids:
            start_min = 0
            for i in range(task_id):
                start_min += self.scratch.daily_schedule[i][1]
            end_min = start_min + self.scratch.daily_schedule[task_id][1]

            start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                          + datetime.timedelta(minutes=start_min))

            end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                        + datetime.timedelta(minutes=end_min))

            start_time_str = start_time.strftime("%H:%M:%S")
            end_time_str = end_time.strftime("%H:%M:%S")
            taks_name = self.scratch.daily_schedule[task_id][0]

            summary = f"{start_time_str} ~ {end_time_str}, {self.name} is planning on {taks_name}"
            task_sumaries += [summary]

            if task_id == action_index:
                task_start_time = start_time
                task_end_time = end_time

        task_context = f"From {', and '.join(task_sumaries)}."

        start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                      + datetime.timedelta(minutes=start_min))
        end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                    + datetime.timedelta(minutes=end_min))

        return task_decomposition_chain.run(agent_name=self.name,
                                                          agent_identity=self.scratch.identity,
                                                          today=self.scratch.time.today,
                                                          task_context=task_context,
                                                          task_description=action_description,
                                                          task_duration=action_duration,
                                                          task_start_time=task_start_time,
                                                          task_end_time=task_end_time
                                                          )

    def _long_term_planning(self, daytype: DayType):
        """
        Formulates the persona's daily long-term plan if it is the start of a new 
        day. This basically has two components: first, we create the wake-up hour, 
        and second, we create the hourly schedule based on it. 
        INPUT
            new_day: Indicates whether the current time signals a "First day",
                    "New day", or False (for neither). This is important because we
                    create the personas' long term planning on the new day. 
        """
        # We start by creating the wake up hour for the persona.
        wake_up_hour = wake_up_hour_chain.run(agent_name=self.name,
                                              agent_lifestyle=self.scratch.lifestyle,
                                              agent_identity=self.scratch.identity)

        # When it is a new day, we start by creating the daily_req of the persona.
        # Note that the daily_req is a list of strings that describe the persona's
        # day in broad strokes.
        if daytype == DayType.FIRST_DAY:
            # Bootstrapping the daily plan for the start of then generation:
            # if this is the start of generation (so there is no previous day's
            # daily requirement, or if we are on a new day, we want to create a new
            # set of daily requirements.

            # TODO parse this to a list
            self.scratch.daily_requirements = first_daily_plan_chain.run(agent_name=self.name,
                                                                         identity=self.scratch.identity,
                                                                         agent_lifestyle=self.scratch.lifestyle,
                                                                         current_day=self.scratch.time.today,
                                                                         wake_up_hour=wake_up_hour)
        elif daytype == DayType.NEW_DAY:
            # TODO parse daily_plan to a list
            daily_plan, current_status = self._generate_daily_plan_and_current_status()

            self.scratch.daily_requirements = daily_plan
            self.scratch.current_status = current_status

        # Based on the daily_req, we create an hourly schedule for the persona,
        # which is a list of todo items with a time duration (in minutes) that
        # add up to 24 hours.

        # TODO parse output properly
        self.scratch.daily_schedule = HourlyBreakdown(identity=self.scratch.identity,
                                                      current_hour=self.scratch.time.hour,
                                                      name=self.name,
                                                      today=self.scratch.time.today,
                                                      hourly_organized_activities=self.scratch.daily_requirements,
                                                      actual_activities=self.scratch.hourly_activity_history,
                                                      current_status=self.scratch.current_status).run()

        # TODO see where this is used
        # complete nonsense so far.. :D
        # persona.scratch.f_daily_schedule_hourly_org = (persona.scratch
        #                                                .f_daily_schedule[:])

        daily_plan = " ,".join(self.scratch.daily_requirements)
        description = f"This is {self.name}'s plan for {self.scratch.time}: {daily_plan}."

        perceived_plan = PerceivedEvent(event_type=EventType.PLAN,
                                        poignancy=0.5,
                                        description=description,
                                        subject=self.name,
                                        predicate="plan",
                                        object_=self.scratch.time.today,
                                        created=self.scratch.time,
                                        expiration=self.scratch.time + datetime.timedelta(days=30))

        self.associative_memory.add(perceived_plan)

    def _generate_daily_plan_and_current_status(self) -> Tuple[str, str]:

        retrieved_events = self._get_related_to_text(
            f"{self.name}'s plan for {self.scratch.time.as_string()}.", EventType.PLAN)
        retrieved_events += self._get_related_to_text(
            f"Important recent events for {self.name}'s life.")

        statements = "[Statements]\n"

        for retrieved_event in retrieved_events:
            statements += f"{retrieved_event.created.strftime('%A %B %d -- %H:%M %p')}: {retrieved_event.description}\n"
        # "name", "today", "yesterday", "statements", "today", "current_activity"

        daily_plan_and_status = daily_plan_and_status_chain.run(agent_name=self.name,
                                                                today=self.scratch.time.today,
                                                                yesterday=self.scratch.time.yesterday,
                                                                statements=self.scratch.associative_memory.latest_events_summary,
                                                                current_activity=self.scratch.current_activity)

        return daily_plan_and_status["daily_plan"], daily_plan_and_status["currently"]

    def _get_related_to_text(self, text: str,  event_type: EventType = None):
        if event_type:
            memories = database.get_by_type(self.name, text, event_type.value)
        else:
            memories = database.get(self.name, text)

        return [PerceivedEvent.from_db_entry(memory.text, memory.metadata) for memory in memories]

    def _get_related_events(self, event: PerceivedEvent, event_type: EventType = None):
        return self._get_related_to_text(event.description, event_type)
