
import datetime
from enum import Enum
from functools import lru_cache
import random
from generative_agents.utils import get_time_string, timeit
from haystack import component

from generative_agents.conversational.pipelines.poignance import rate_poignance

from generative_agents.core.events import Action, Event, EventType, ObjectAction, PerceivedEvent
from generative_agents.core.whisper.whisper import whisper
from generative_agents.persistence.database import ConversationFilling
from generative_agents.simulation.maze import Level
from generative_agents.simulation.time import DayType
from generative_agents.persistence import database


from generative_agents.conversational.pipelines.wake_up_hour import estimate_wake_up_hour
from generative_agents.conversational.pipelines.daily_plan import create_daily_plan_and_status
from generative_agents.conversational.pipelines.hourly_breakdown import create_hourly_schedule
from generative_agents.conversational.pipelines.contextualize_event import contextualize_event
from generative_agents.conversational.pipelines.decide_to_talk import decide_to_talk
from generative_agents.conversational.pipelines.decide_to_react import decide_to_react
from generative_agents.conversational.pipelines.action_location_sector import action_sector_locations
from generative_agents.conversational.pipelines.action_location_arena import action_area_locations
from generative_agents.conversational.pipelines.action_location_game_object import action_location_game_object
from generative_agents.conversational.pipelines.object_event import describe_object_state
from generative_agents.conversational.pipelines.action_pronunciatio import action_pronunciatio
from generative_agents.conversational.pipelines.action_event_tripple import action_event_triple
from generative_agents.conversational.pipelines.summarize_chat_relationship import summarize_chat_relationship
from generative_agents.conversational.pipelines.conversation import run_conversation
from generative_agents.conversational.pipelines.conversation_summary import conversation_summary

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.conversational.pipelines.new_decomposition_schedule import create_new_decomposition_schedule
from generative_agents.conversational.pipelines.task_decomposition import create_decomposition_schedule
from generative_agents.conversational.pipelines.first_daily_plan import create_daily_plan


class ReactionMode(Enum):
    CHAT = "chat"
    WAIT = "wait"
    DO_OTHER_THINGS = "do other things"
    
@component
class Plan:
    def __init__(self, agent):
        self.agent = agent

    @timeit
    @component.output_types(address=str)
    def run(self, agents: dict[str, 'Agent'], daytype: DayType, retrieved: dict[str, dict[str, list[PerceivedEvent]]]) -> str:
        if daytype == DayType.NEW_DAY or daytype == DayType.FIRST_DAY:
            whisper(self.agent.name, f"planning first daily plan")
            self._long_term_planning(daytype)

        if self.agent.scratch.is_action_finished():
            self._determine_action()
            whisper(
                self.agent.name, f"planning to {self.agent.scratch.action.event.description}")

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
            if focused_event:
                whisper(
                    self.agent.name, f"focusing on {focused_event['curr_event'].description}")
            else:
                whisper(self.agent.name, f"not focusing on any persona event")

        # Step 2: Once we choose an event, we need to determine whether the
        #         persona will take any actions for the perceived event. There are
        #         three possible modes of reaction returned by _should_react.
        #         a) "chat with {target_persona.name}"
        #         b) "react"
        #         c) False

        if self.agent.scratch.action.event.predicate == "chat with":
            last_utterance = self.agent.scratch.action.event.filling[-1]
            if not last_utterance.end and last_utterance.name != self.agent.name:
                self._chat_react(agent_with=agents[last_utterance.name])
                focused_event = None

        if focused_event:
            reaction_mode, payload = self._should_react(focused_event, agents)
            whisper(
                self.agent.name, f"reaction mode is {reaction_mode} with payload {payload}")
            if reaction_mode and reaction_mode != ReactionMode.DO_OTHER_THINGS:
                # If we do want to chat, then we generate conversation
                if reaction_mode == ReactionMode.CHAT:
                    self._chat_react(agent_with=payload)
                elif reaction_mode == ReactionMode.WAIT:
                    self._wait_react(payload)
                # elif reaction_mode == "do other things":
                #   _chat_react(persona, focused_event, reaction_mode, personas)

        # Step 3: Chat-related state clean up.
        # If the persona is not chatting with anyone, we clean up any of the
        # chat-related states here.

        if self.agent.scratch.action.event.predicate != "chat with":
            self.agent.scratch.chatting_with = None
            self.agent.scratch.chat = None
            self.agent.scratch.chatting_end_time = None

        # We want to make sure that the persona does not keep conversing with each
        # other in an infinite loop. So, chatting_with_buffer maintains a form of
        # buffer that makes the persona wait from talking to the same target
        # immediately after chatting once. We keep track of the buffer value here.
        curr_persona_chat_buffer = self.agent.scratch.chatting_with_buffer
        for persona_name, _ in curr_persona_chat_buffer.items():
            if persona_name != self.agent.scratch.chatting_with:
                self.agent.scratch.chatting_with_buffer[persona_name] -= 1

        return {"address": self.agent.scratch.action.address}

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
        wake_up_hour = estimate_wake_up_hour(
            self.agent.name, self.agent.scratch.identity, self.agent.scratch.lifestyle)

        whisper(self.agent.name, f"wake up hour is at {wake_up_hour}")

        # When it is a new day, we start by creating the daily_req of the persona.
        # Note that the daily_req is a list of strings that describe the persona's
        # day in broad strokes.
        if daytype == DayType.FIRST_DAY:
            self.agent.scratch.daily_requirements = create_daily_plan(self.agent.scratch.identity,
                                                                              self.agent.scratch.time.today,
                                                                              self.agent.name,
                                                                              wake_up_hour)


        if daytype == DayType.NEW_DAY:
            daily_plan, current_status = self._generate_daily_plan_and_current_status()

            whisper(self.agent.name, f"new daily plan is {daily_plan}")
            whisper(self.agent.name, f"new current status is {current_status}")

            self.agent.scratch.daily_requirements = daily_plan
            self.agent.scratch.current_status = current_status

        # Based on the daily_req, we create an hourly schedule for the persona,
        # which is a list of todo items with a time duration (in minutes) that
        # add up to 24 hours.
        self.agent.scratch.daily_schedule = create_hourly_schedule(self.agent.name,
                                                                   self.agent.scratch.identity,
                                                                   self.agent.scratch.daily_requirements,
                                                                   wake_up_hour)

        daily_plan = ",".join([entry['activity']
                              for entry in self.agent.scratch.daily_schedule])
        description = f"This is {self.agent.name}'s plan for {self.agent.scratch.time.today}: {daily_plan}."
        self.agent.scratch.daily_schedule_hourly_organzied = self.agent.scratch.daily_schedule = [
            (entry['activity'], 60) for entry in self.agent.scratch.daily_schedule]

        whisper(self.agent.name, f"new daily plan is {description}")

        perceived_plan = PerceivedEvent(event_type=EventType.PLAN,
                                        poignancy=0.5,
                                        depth=1,
                                        description=description,
                                        subject=self.agent.name,
                                        predicate="plan",
                                        object_=self.agent.scratch.time.today,
                                        created=self.agent.scratch.time.time,
                                        expiration=self.agent.scratch.time.time +
                                        datetime.timedelta(days=30),
                                        tile=self.agent.scratch.tile)

        self.agent.associative_memory.add(perceived_plan)

    def _generate_daily_plan_and_current_status(self):
        retrieved_events = self._get_related_to_text(
            f"{self.agent.name}'s plan for {self.agent.scratch.time.as_string()}.", EventType.PLAN)
        retrieved_events += self._get_related_to_text(
            f"Important recent events for {self.agent.name}'s life.")

        statements = "[Statements]\n"

        for retrieved_event in retrieved_events:
            statements += f"{retrieved_event.created.strftime('%A %B %d -- %H:%M %p')}: {retrieved_event.description}\n"
        
        return create_daily_plan_and_status(self.agent.name,
                                        self.agent.scratch.identity,
                                        self.agent.scratch.time.today,
                                        self.agent.scratch.time.yesterday,
                                        statements,
                                        self.agent.scratch.current_activity)

    def _get_related_to_text(self, text: str,  event_type: EventType = None):
        if event_type:
            memories = database.get_by_type(self.agent.name, text, event_type.value)
        else:
            memories = database.get(self.agent.name, text)

        return [PerceivedEvent.from_db_entry(memory.text, memory.metadata) for memory in memories]

    def _get_related_events(self, event: PerceivedEvent, event_type: EventType = None):
        return self._get_related_to_text(event.description, event_type)
    
    def _determine_action(self):

        def needs_decomposition(action_description: str, action_duration: int):
            # TODO reformulate this logic

            desc = action_description.lower()

            if "sleep" not in desc and "bed" not in desc:
                return True
            elif "sleeping" in desc or "asleep" in desc or "in bed" in desc:
                return False
            elif "sleep" in desc or "bed" in desc:
                if action_duration > 60:
                    return False
                return True

        current_index = self.agent.scratch.get_daily_schedule_index()
        next_hour_index = self.agent.scratch.get_daily_schedule_index(
            advance=60)

        # * Decompose *
        # During the first hour of the day, we need to decompose two hours
        # sequence. We do that here.
        if current_index == 0:
            # This portion is invoked if it is the first hour of the day.
            action_description, action_duration = self.agent.scratch.daily_schedule[
                current_index]

            if action_duration >= 60:
                # We decompose if the next action is longer than an hour, and fits the
                # criteria described in determine_decomp.
                if needs_decomposition(action_description, action_duration):
                    self.agent.scratch.daily_schedule[current_index:current_index +
                                                      1] = self._decompose_action(current_index, action_description, action_duration)

            if next_hour_index + 1 < len(self.agent.scratch.daily_schedule):
                action_description, action_duration = self.agent.scratch.daily_schedule[
                    next_hour_index]

                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        self.agent.scratch.daily_schedule[next_hour_index:next_hour_index + 1] = self._decompose_action(
                            next_hour_index, action_description, action_duration)

        if next_hour_index < len(self.agent.scratch.daily_schedule):
            # If it is not the first hour of the day, this is always invoked (it is
            # also invoked during the first hour of the day -- to double up so we can
            # decompose two hours in one go). Of course, we need to have something to
            # decompose as well, so we check for that too.
            if self.agent.scratch.time.time.hour < 23:
                # And we don't want to decompose after 11 pm.
                action_description, action_duration = self.agent.scratch.daily_schedule[
                    next_hour_index]
                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        # current_index:next_hour_index
                        decomposition = self._decompose_action(next_hour_index, action_description, action_duration)
                        self.agent.scratch.daily_schedule[next_hour_index:next_hour_index+1] = (
                            decomposition)

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

        action_description, action_duration = self.agent.scratch.daily_schedule[current_index]

        action_game_object = None
        next_address = ""
        whisper(
            self.agent.name, f"determined next action: {action_description}")
        action_sector = self._generate_next_action_sector(action_description)

        whisper(self.agent.name,
                f"determined next sector: {action_sector}")
        action_arena = self._generate_next_action_arena(
            action_description, action_sector)
        whisper(self.agent.name, f"determined next arena: {action_arena}")
        next_address = self._generate_next_action_game_object(
            action_description, action_arena)

        address_parts = next_address.split(":")
        tile = self.agent.spatial_memory[address_parts[0]][address_parts[1]
                                                            ][address_parts[2]].game_objects[address_parts[3]]

        whisper(self.agent.name,
                f"determined next game object: {action_game_object}")

        action_pronouncio = self._generate_action_pronunciatio(
            action_description)
        whisper(self.agent.name,
                f"determined next pronouncio: {action_pronouncio}")

        action_event = self._generate_action_event_triple(action_description)
        whisper(self.agent.name,
                f"determined next event triple: {action_event}")

        object_action = None
        if next_address != "<random>":
            action_object_desctiption, tripplet = self._generate_action_object_description(
                next_address, action_description)
            whisper(
                self.agent.name, f"determined next object description: {action_object_desctiption}")
            action_object_pronunciatio = self._generate_action_pronunciatio(
                action_object_desctiption)
            whisper(
                self.agent.name, f"determined next object pronouncio: {action_object_pronunciatio}")
            subject, predicate, object_ = tripplet
            whisper(
                self.agent.name, f"determined next object event triple: {subject}, {predicate}, {object_}")

            object_action = ObjectAction(address=next_address,
                                         emoji=action_object_pronunciatio,
                                         event=Event(subject=subject,
                                                     predicate=predicate,
                                                     object_=object_,
                                                     description=action_object_desctiption,
                                                     depth=0,
                                                     tile=tile))

        minutes_from_now = self.agent.time.time.hour * 60 + self.agent.time.time.minute

        planned_end = 0
        for _, duration in self.agent.scratch.daily_schedule[:current_index+1]:
            planned_end += int(duration)

        minutes_left = planned_end - minutes_from_now + 1

        next_action = Action(address=next_address,
                             start_time=self.agent.scratch.time.time,
                             duration=minutes_left,
                             emoji=action_pronouncio,
                             event=Event(subject=self.agent.name,
                                         predicate=action_event[1],
                                         object_=action_event[2],
                                         description=action_description,
                                         depth=0,
                                         tile=tile
                                         ),
                             object_action=object_action)

        if self.agent.scratch.action:
            self.agent.scratch.finished_action.append(
                self.agent.scratch.action)
        self.agent.scratch.action = next_action

    def _should_react(self, focused_event: dict[str, list[PerceivedEvent]], agents: list[str, 'Agent']):
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
        def lets_talk(init_agent: 'Agent', target_agent: 'Agent', retrieved: dict[str, list[PerceivedEvent]]):
            if init_agent.name == target_agent.name:
                return False

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

            if self._generate_decide_to_talk(target_agent, retrieved):
                return True

            return False

        def lets_react(init_agent: 'Agent', target_agent: 'Agent', retrieved):
            if (not target_agent.scratch.action.address
                or not target_agent.scratch.action.event.description
                or not init_agent.scratch.action.address
                    or not init_agent.scratch.action.event.description):
                return ReactionMode.DO_OTHER_THINGS, ModuleNotFoundError

            if ("sleeping" in target_agent.scratch.action.event.description
                    or "sleeping" in init_agent.scratch.action.event.description):
                return ReactionMode.DO_OTHER_THINGS, None

            if init_agent.scratch.time.hour == 23:
                return ReactionMode.DO_OTHER_THINGS, None

            if "waiting" in target_agent.scratch.action.event.description:
                return ReactionMode.DO_OTHER_THINGS, None

            if not init_agent.scratch.planned_path:
                return ReactionMode.DO_OTHER_THINGS, None

            if (init_agent.scratch.action.address.split(":")[:-1] != target_agent.scratch.action.address.split(":")[:-1] or
                    init_agent.scratch.tile.l2_distance(target_agent.scratch.tile) >= 4):
                return ReactionMode.DO_OTHER_THINGS, None

            react_mode = self.agent._generate_decide_to_react(
                target_agent, retrieved)

            if react_mode == 1:
                wait_until = ((target_agent.scratch.action.start_time
                               + datetime.timedelta(minutes=target_agent.scratch.action.duration - 1))
                              .strftime("%B %d, %Y, %H:%M:%S"))
                return ReactionMode.WAIT, wait_until
            elif react_mode == 2:
                return ReactionMode.DO_OTHER_THINGS, None
            else:
                return ReactionMode.DO_OTHER_THINGS, None

        # If the persona is chatting right now, default to no reaction
        if self.agent.scratch.chatting_with:
            return ReactionMode.DO_OTHER_THINGS, None
        if "<waiting>" in self.agent.scratch.action.address:
            return ReactionMode.WAIT, None

        # Recall that retrieved takes the following form:
        # dictionary {["curr_event"] = <ConceptNode>,
        #             ["events"] = [<ConceptNode>, ...],
        #             ["thoughts"] = [<ConceptNode>, ...]}
        curr_event = focused_event["curr_event"]

        if ":" not in curr_event.subject and curr_event.subject in agents:
            # this is a persona event.
            # TODO error handling
            target_agent = agents[curr_event.subject]

            if (target_agent.name in self.agent.scratch.chatting_with_buffer):
                if self.agent.scratch.chatting_with_buffer[target_agent.name] > 0:
                    return ReactionMode.DO_OTHER_THINGS, None

            if lets_talk(self, agents[curr_event.subject], focused_event):
                return ReactionMode.CHAT, target_agent

            return lets_react(self, agents[curr_event.subject], focused_event)

        return ReactionMode.DO_OTHER_THINGS, None

    def _chat_react(self, agent_with: 'Agent'):
        utterance, end = self._generate_conversation(agent_with)
        conversation = self.agent.associative_memory.active_conversation_with(
            agent_with.name)

        action_start_time = agent_with.scratch.time.time
        filling = []

        if conversation:
            filling = conversation.filling
            action_start_time = conversation.created

        filling += [ConversationFilling(name=self.agent.name,
                                        utterance=utterance, end=end)]
        description = self._generate_conversation_summary(filling)

        self._create_react_action(inserted_action=description,
                                  inserted_action_duration=10,
                                  action_address=f"<persona> {agent_with.name}",
                                  action_event=(
                                      self.agent.name, "chat with", agent_with.name),
                                  chatting_with=agent_with.name,
                                  chat=utterance,
                                  chatting_with_buffer={
                                      agent_with.name: 800},
                                  chatting_end_time=None,
                                  action_pronunciatio="💬",
                                  filling=filling,
                                  action_start_time=action_start_time)

        agent_with._create_react_action(inserted_action=description,
                                        inserted_action_duration=10,
                                        action_address=f"<persona> {self.agent.name}",
                                        action_event=(
                                            agent_with.name, "chat with", self.agent.name),
                                        chatting_with=agent_with.name,
                                        chat="-",
                                        chatting_with_buffer={
                                            agent_with.name: 800},
                                        chatting_end_time=None,
                                        action_pronunciatio="💬",
                                        filling=filling,
                                        action_start_time=action_start_time)

        if end:
            duration_minutes = round(
                (self.agent.scratch.time.time - self.agent.scratch.action.start_time).total_seconds() / 60)
            self._update_schedule(description, duration_minutes)
            self.agent.scratch.chatting_with = None
            self.agent.scratch.chat = None
            self.agent.scratch.chatting_end_time = self.agent.scratch.time

    def _wait_react(self, wait_time):
        event_short_description = self.agent.scratch.action.event.description.split(
            "(")[-1][:-1]

        inserted_action = f'waiting to start {event_short_description}'
        end_time = datetime.strptime(wait_time, "%B %d, %Y, %H:%M:%S")
        inserted_action_duration = round((end_time.minute + end_time.hour * 60) - (
            self.agent.scratch.time.time.minute + self.agent.scratch.time.time.hour * 60) + 1)

        action_address = f"<waiting> {self.agent.scratch.tile.x} {self.agent.scratch.tile.y}"
        action_event = (self.agent.name, "waiting to start",
                        event_short_description)
        chatting_with = None
        chat = None
        chatting_with_buffer = None
        chatting_end_time = None

        action_pronunciatio = "⌛"

        self.agent._update_schedule(inserted_action, inserted_action_duration)
        self.agent._create_react_action(inserted_action, inserted_action_duration,
                                        action_address, action_event, chatting_with, chat, chatting_with_buffer, chatting_end_time,
                                        action_pronunciatio)

    async def _generate_decide_to_talk(self, target_agent: 'Agent', retrieved: dict[str, list[PerceivedEvent]]):
        context = contextualize_event(agent=self.agent.name,
                                      identity=self.agent.scratch.identity,
                                      event_description=retrieved["curr_event"].description,
                                      events=", ".join(
                                          [event.description for event in retrieved["events"]]),
                                      thoughts=", ".join([thought.description for thought in retrieved["thoughts"]]))

        last_conversation = self.agent.associative_memory.last_conversation_with(
            target_agent.name)

        if last_conversation:
            last_chat_summary = f"last chatted at {last_conversation.created.strftime('%B %d, %Y, %H:%M:%S')} about {last_conversation.description}"
        else:
            last_chat_summary = "never chatted before."
        return decide_to_talk(context=context,
                              current_time=self.agent.scratch.time.as_string(),
                              init_agent=self.agent.name,
                              init_agent_observation=self.agent.scratch.action.event.description,
                              agent_with=target_agent.name,
                              agent_with_observation=retrieved["curr_event"].description,
                              last_chat_summary=last_chat_summary)

    def _generate_decide_to_react(self, target_agent: 'Agent', retrieved: dict[str, list[PerceivedEvent]]):
        context = self._focused_event_to_context(retrieved)
        current_time = self.agent.scratch.time.as_string()

        return decide_to_react(context=context,
                               current_time=current_time,
                               agent=self.agent.name,
                               agent_with=target_agent.name,
                               agent_with_action=target_agent.scratch.action.event.description,
                               agent_observation=self.observation,
                               agent_with_observation=target_agent.observation,
                               initial_action_description=self.agent.scratch.action.event.description)

    def _generate_next_action_sector(self, action_description):
        name = self.agent.name
        home = self.agent.scratch.home.sector
        home_arenas = self.agent.spatial_memory.get_str_accessible_sector_arenas(
            self.agent.scratch.home.get_path(Level.SECTOR))
        current_sector = self.agent.scratch.tile.sector
        current_sector_arenas = self.agent.spatial_memory.get_str_accessible_sector_arenas(
            self.agent.scratch.tile.get_path(Level.SECTOR))
        nearby_sectors = self.agent.spatial_memory.get_str_accessible_sectors(
            self.agent.scratch.tile.get_path(Level.WORLD))

        next_sector = action_sector_locations(agent_name=name,
                                              agent_home=home,
                                              agent_home_arenas=home_arenas,
                                              agent_current_sector=current_sector,
                                              agent_current_sector_arenas=current_sector_arenas,
                                              available_sectors_nearby=nearby_sectors,
                                              curr_action_description=action_description)

        return f"{self.agent.scratch.tile.world}:{next_sector}"

    def _generate_next_action_arena(self, action_description, action_sector):
        name = self.agent.name
        current_sector = self.agent.scratch.tile.sector
        current_area = self.agent.scratch.tile.arena
        sector = action_sector.split(":")[-1]

        arena = action_area_locations(name=name,
                                      current_area=current_area,
                                      current_sector=current_sector,
                                      sector=sector,
                                      sector_arenas=self.agent.spatial_memory.get_str_accessible_sector_arenas(
                                          action_sector),
                                      action_description=action_description)

        return f"{action_sector}:{arena}"

    def _generate_next_action_game_object(self, action_description, action_arena):
        arena = action_arena
        available_objects = self.agent.spatial_memory.get_str_accessible_arena_game_objects(
            action_arena)

        if not available_objects:
            arena = self.agent.scratch.tile.get_path(Level.ARENA)
            available_objects = self.agent.spatial_memory.get_str_accessible_arena_game_objects(
                arena)

        game_object = action_location_game_object(action_description=action_description,
                                                  available_objects=available_objects)

        return f"{arena}:{game_object}"

    def _generate_action_object_description(self, action_game_object, action_description):

        object_name = action_game_object.split(":")[-1]
        return describe_object_state(name=self.agent.name,
                                     object_name=object_name,
                                     object_address=action_game_object,
                                     action_description=action_description)

    def _generate_action_pronunciatio(self, action_description):
        return action_pronunciatio(action_description)
    
    def _generate_action_event_triple(self, action_description):
        return action_event_triple(self.agent.name, action_description)
    
    def _choose_retrieved(self, retrieved: dict[str, dict[str, list[PerceivedEvent]]]):
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
        no_self_event_retrieved = {description: context for description, context in retrieved.items(
        ) if context["curr_event"].subject != self.agent.name}

        persona_context = [context for _, context in no_self_event_retrieved.items(
        ) if ":" not in context["curr_event"].subject]
        if persona_context:
            return random.choice(persona_context)

        non_idle_context = [context for _, context in no_self_event_retrieved.items(
        ) if "idle" not in context["curr_event"].description]
        if non_idle_context:
            return random.choice(non_idle_context)

        return None
    
    def _generate_conversation(self, agent_with: 'Agent'):
        retrieved = self.agent.associative_memory.retrieve_relevant_entries(
            [agent_with.name], 50)
        relationship = self._generate_summarize_agent_relationship(agent_with, retrieved)

        focal_points = [
            f"{relationship}", f"{agent_with.name} is {agent_with.scratch.action.event.description}"]
        active_conversation = self.agent.associative_memory.active_conversation_with(
            agent_with.name)

        active_conversation_string = ""
        if active_conversation:
            active_conversation_string = "\n".join(
                [f"{filling.name}: {filling.utterance}" for filling in active_conversation.filling])
            focal_points += [active_conversation_string]
        else:
            active_conversation = ""

        retrieved += self.agent.associative_memory.retrieve_relevant_entries(
            focal_points, 15)
        location = f"{self.agent.scratch.tile.arena} in {self.agent.scratch.tile.sector}"
        description = list(set([event.description for event in retrieved]))
        memory = "\n".join(description)

        last_conversation = self.agent.associative_memory.last_conversation_with(
            agent_with.name)

        past_context = ""

        if last_conversation and (previous_chat := int((self.agent.scratch.time.time - last_conversation.created).total_seconds()/60)) > 480:
            past_context = f"{str(previous_chat)} minutes ago, {self.agent.name} and {agent_with.name} were already {last_conversation.description}. This context takes place after that conversation."

        return run_conversation(identity=self.agent.scratch.identity,
                                agent=self.agent.name,
                                agent_with=agent_with.name,
                                agent_with_action=agent_with.scratch.action.event.description,
                                agent_action=self.agent.scratch.action.event.description,
                                location=location,
                                conversation=active_conversation_string,
                                memory=memory,
                                past_context=past_context)
    

    def _generate_summarize_agent_relationship(self, agent_with: 'Agent', retrieved: list[PerceivedEvent]):
        """
        We generate the relationship between the two agents. 
        """
        statements = "\n".join([event.description for event in retrieved])

        return summarize_chat_relationship(agent=self.agent.name,
                                           agent_with=agent_with.name,
                                           statements=statements)
    

    def _generate_conversation_summary(self, conversation_filling: list[ConversationFilling]):
        conversation_history = "\n".join(
            [f"{filling.name}: {filling.utterance}" for filling in conversation_filling])
        return conversation_summary(conversation_history=conversation_history)
    

    def _create_react_action(self, inserted_action, inserted_action_duration,
                             action_address, action_event, chatting_with, chat, chatting_with_buffer,
                             chatting_end_time, action_pronunciatio, filling=[], action_start_time=None):

        event_poignancy = self._rate_perception_poignancy(EventType.CHAT, inserted_action)
        
        event = PerceivedEvent(depth=0,
                      subject=self.agent.name,
                      predicate=action_event[1],
                      object_=action_event[2],
                      description=inserted_action,
                      filling=filling,
                      poignancy=event_poignancy,
                      event_type=EventType.CHAT,
                      tile=self.agent.scratch.tile)
        
        next_action = Action(address=action_address,
                             start_time=action_start_time,
                             duration=inserted_action_duration,
                             emoji=action_pronunciatio,
                             event=event)

        self.agent.scratch.finished_action.append(self.agent.scratch.action)
        self.agent.scratch.action = next_action
        self.agent.scratch.chatting_with = chatting_with
        self.agent.scratch.chat = chat
        if chatting_with_buffer:
            self.agent.scratch.chatting_with_buffer = {**self.agent.scratch.chatting_with_buffer, **chatting_with_buffer}
        self.agent.scratch.chatting_end_time = chatting_end_time
        self.associative_memory.add(event)

    def _rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = rate_poignance(self.agent.name, self.agent.scratch.identity, event_type.value, description)

        # TODO properly check the output here
        return int(score) / 10
    
    def _update_schedule(self, inserted_action, inserted_action_duration):
        start_hour, end_hour = self.__calculate_start_end_hours()
        start_index, end_index = self.__get_start_and_end_index(start_hour, end_hour)

        new_schedule = create_new_decomposition_schedule(agent=self.agent.name,
                                                          start_hour=start_hour,
                                                          end_hour=end_hour,
                                                          new_event=inserted_action,
                                                          new_event_duration=inserted_action_duration,
                                                          new_event_index=self.agent.scratch.get_daily_schedule_index() - start_index,
                                                          schedule_slice=self.agent.scratch.daily_schedule_hourly_organzied[start_index:end_index])

        self.agent.scratch.daily_schedule[start_index:end_index] = new_schedule


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
        current_index = self.agent.scratch.get_daily_schedule_index()

        task_ids = []
        # if curr_f_org_index > 0:
        #   all_indices += [curr_f_org_index-1]
        task_ids += [current_index]
        if current_index+1 <= len(self.agent.scratch.daily_schedule):
            task_ids += [current_index+1]
        if current_index+2 < len(self.agent.scratch.daily_schedule):
            task_ids += [current_index+2]

        task_sumaries = []
        for task_id in task_ids:
            start_min = 0
            for i in range(task_id):
                start_min += self.agent.scratch.daily_schedule[i][1]
            end_min = start_min + self.agent.scratch.daily_schedule[task_id][1]

            start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                          + datetime.timedelta(minutes=start_min))

            end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                        + datetime.timedelta(minutes=end_min))

            start_time_str = start_time.strftime("%H:%M:%S")
            end_time_str = end_time.strftime("%H:%M:%S")
            taks_name = self.agent.scratch.daily_schedule[task_id][0]

            summary = f"{start_time_str} ~ {end_time_str}, {self.agent.name} is planning on {taks_name}"
            task_sumaries += [summary]

        task_context = f"From {', and '.join(task_sumaries)}."

        start_time = get_time_string(datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                      + datetime.timedelta(minutes=start_min))
        end_time = get_time_string(datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                    + datetime.timedelta(minutes=end_min))

        return create_decomposition_schedule(name=self.agent.name,
                                            identity=self.agent.scratch.identity,
                                            today=self.agent.scratch.time.today,
                                            task_context=task_context,
                                            task_description=action_description,
                                            task_duration=action_duration,
                                            task_start_time=start_time,
                                            task_end_time=end_time)

    