


from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import math
from operator import itemgetter
from haystack import Pipeline, component

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.core.agent import ReactionMode
from generative_agents.core.events import Event, EventType, PerceivedEvent
from generative_agents.core.whisper.whisper import whisper
from generative_agents.simulation.maze import Level, Maze
from generative_agents.simulation.time import DayType

from generative_agents.conversational.pipelines.wake_up_hour import estimate_wake_up_hour

@component
class Plan:
    def __init__(self, agent):
        self.agent = agent

    @component.output_types(address=str)
    def run(self, agents: dict[str, 'Agent'], daytype: DayType, retrieved: dict[str, dict[str, list[PerceivedEvent]]]):
        if daytype == DayType.NEW_DAY or daytype == DayType.FIRST_DAY:
            whisper(self.agent.name, f"planning first daily plan")
            self._long_term_planning(daytype)
        

        if self.agent.scratch.is_action_finished():
            self.agent._determine_action()
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
                self.agent._chat_react(agent_with=agents[last_utterance.name])
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

        return self.agent.scratch.action.address
    

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
        wake_up_hour = estimate_wake_up_hour(self.agent.name, self.agent.scratch.identity, self.agent.scratch.lifestyle)


        whisper(self.agent.name, f"wake up hour is at {wake_up_hour}")

        # When it is a new day, we start by creating the daily_req of the persona.
        # Note that the daily_req is a list of strings that describe the persona's
        # day in broad strokes.
        if daytype == DayType.FIRST_DAY:
            # Bootstrapping the daily plan for the start of then generation:
            # if this is the start of generation (so there is no previous day's
            # daily requirement, or if we are on a new day, we want to create a new
            # set of daily requirements.
            self.agent.scratch.daily_requirements = await FirstDailyPlan(agent_name=self.agent.name,
                                                                   agent_identity=await self.agent.scratch.identity,
                                                                   agent_lifestyle=self.agent.scratch.lifestyle,
                                                                   current_day=self.agent.scratch.time.today,
                                                                   wake_up_hour=wake_up_hour).run()
            whisper(
                self.agent.name, f"first daily plan is {self.agent.scratch.daily_requirements}")
        elif daytype == DayType.NEW_DAY:
            # TODO parse daily_plan to a list
            daily_plan, current_status = self.agent._generate_daily_plan_and_current_status()

            whisper(self.agent.name, f"new daily plan is {daily_plan}")
            whisper(self.agent.name, f"new current status is {current_status}")

            self.agent.scratch.daily_requirements = daily_plan
            self.agent.scratch.current_status = current_status

        # Based on the daily_req, we create an hourly schedule for the persona,
        # which is a list of todo items with a time duration (in minutes) that
        # add up to 24 hours.

        self.agent.scratch.daily_schedule = await HourlyBreakdown(identity=await self.agent.scratch.identity,
                                                            wake_up_hour=wake_up_hour,
                                                            name=self.agent.name,
                                                            hourly_organized_activities=self.agent.scratch.daily_requirements).run()
        
        daily_plan = ",".join([entry['activity'] for entry in self.agent.scratch.daily_schedule])
        description = f"This is {self.agent.name}'s plan for {self.agent.scratch.time.today}: {daily_plan}."
        self.agent.scratch.daily_schedule_hourly_organzied = self.agent.scratch.daily_schedule = [(entry['activity'], 60) for entry in self.agent.scratch.daily_schedule]

        whisper(self.agent.name, f"new daily plan is {description}")

        perceived_plan = PerceivedEvent(event_type=EventType.PLAN,
                                        poignancy=0.5,
                                        depth=1,
                                        description=description,
                                        subject=self.agent.name,
                                        predicate="plan",
                                        object_=self.agent.scratch.time.today,
                                        created=self.agent.scratch.time.time,
                                        expiration=self.agent.scratch.time.time + datetime.timedelta(days=30),
                                        tile=self.agent.scratch.tile)

        self.agent.associative_memory.add(perceived_plan)

    async def _determine_action(self):

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

        current_index = self.agent.scratch.get_daily_schedule_index()
        next_hour_index = self.agent.scratch.get_daily_schedule_index(advance=60)

        # * Decompose *
        # During the first hour of the day, we need to decompose two hours
        # sequence. We do that here.
        if current_index == 0:
            # This portion is invoked if it is the first hour of the day.
            action_description, action_duration = self.agent.scratch.daily_schedule[current_index]

            if action_duration >= 60:
                # We decompose if the next action is longer than an hour, and fits the
                # criteria described in determine_decomp.
                if needs_decomposition(action_description, action_duration):
                    self.agent.scratch.daily_schedule[current_index:current_index +
                                                1] = await self.agent._decompose_action(current_index, action_description, action_duration)

            if next_hour_index + 1 < len(self.agent.scratch.daily_schedule):
                action_description, action_duration = self.agent.scratch.daily_schedule[next_hour_index]

                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        self.agent.scratch.daily_schedule[next_hour_index:next_hour_index + 1] = await self.agent._decompose_action(
                            next_hour_index, action_description, action_duration)

        if next_hour_index < len(self.agent.scratch.daily_schedule):
            # If it is not the first hour of the day, this is always invoked (it is
            # also invoked during the first hour of the day -- to double up so we can
            # decompose two hours in one go). Of course, we need to have something to
            # decompose as well, so we check for that too.
            if self.agent.scratch.time.time.hour < 23:
                # And we don't want to decompose after 11 pm.
                action_description, action_duration = self.agent.scratch.daily_schedule[next_hour_index]
                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        # current_index:next_hour_index
                        decomposition = await self.agent._decompose_action(next_hour_index, action_description, action_duration)
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
        try:
            whisper(
                self.agent.name, f"determined next action: {action_description}")
            action_sector = await self.agent._generate_next_action_sector(action_description)

            if action_sector == "<random>":
                next_address = action_sector
                tile = None
            else:
                whisper(self.agent.name, f"determined next sector: {action_sector}")
                action_arena = await self.agent._generate_next_action_arena(
                    action_description, action_sector)
                whisper(self.agent.name, f"determined next arena: {action_arena}")
                next_address = await self.agent._generate_next_action_game_object(
                    action_description, action_arena)
                
                address_parts = next_address.split(":")
                tile = self.agent.spatial_memory[address_parts[0]][address_parts[1]][address_parts[2]].game_objects[address_parts[3]]

                whisper(self.agent.name, f"determined next game object: {action_game_object}")
        except Exception as e:
            print(f"Unable to generate next action: {e}")
            next_address = "<random>"
            tile = None

        action_pronouncio = await self.agent._generate_action_pronunciatio(
            action_description)
        whisper(self.agent.name, f"determined next pronouncio: {action_pronouncio}")

        action_event = await self.agent._generate_action_event_triple(action_description)
        whisper(self.agent.name, f"determined next event triple: {action_event}")

        object_action = None
        if next_address != "<random>":
            action_object_desctiption, tripplet = await self.agent._generate_action_object_description(
                next_address, action_description)
            whisper(self.agent.name, f"determined next object description: {action_object_desctiption}")
            action_object_pronunciatio = await self.agent._generate_action_pronunciatio(
                action_object_desctiption)
            whisper(self.agent.name, f"determined next object pronouncio: {action_object_pronunciatio}")
            subject, predicate, object_ = tripplet
            whisper(self.agent.name, f"determined next object event triple: {subject}, {predicate}, {object_}")

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
            self.agent.scratch.finished_action_queue.put(self.agent.scratch.action)
        self.agent.scratch.action = next_action

    async def _chat_react(self, agent_with: 'Agent'):
        utterance, end = await self.agent._generate_conversation(agent_with)
        conversation = self.agent.associative_memory.active_conversation_with(
            agent_with.name)

        action_start_time = agent_with.scratch.time.time
        filling = []

        if conversation:
            filling = conversation.filling
            action_start_time = conversation.created

        filling += [ConversationFilling(name=self.agent.name, utterance=utterance, end=end)]
        description = await self.agent._generate_conversation_summary(filling)

        await self.agent._create_react_action(inserted_action=description,
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

        await agent_with._create_react_action(inserted_action=description,
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
            duration_minutes = round((self.agent.scratch.time.time - self.agent.scratch.action.start_time).total_seconds() / 60)
            await self.agent._update_schedule(description, duration_minutes)
            self.agent.scratch.chatting_with = None
            self.agent.scratch.chat = None
            self.agent.scratch.chatting_end_time = self.agent.scratch.time


    async def _should_react(self, focused_event: Dict[str, List[PerceivedEvent]], agents: Dict[str, 'Agent']):
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
        async def lets_talk(init_agent: Agent, target_agent: Agent, retrieved: Dict[str, List[PerceivedEvent]]):
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

            if await self.agent._generate_decide_to_talk(target_agent, retrieved):
                return True

            return False

        async def lets_react(init_agent: 'Agent', target_agent: 'Agent', retrieved):
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

            react_mode = await self.agent._generate_decide_to_react(target_agent, retrieved)

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

            if await lets_talk(self, agents[curr_event.subject], focused_event):
                return ReactionMode.CHAT, target_agent

            return await lets_react(self, agents[curr_event.subject], focused_event)

        return ReactionMode.DO_OTHER_THINGS, None

    async def _chat_react(self, agent_with: 'Agent'):
        utterance, end = await self.agent._generate_conversation(agent_with)
        conversation = self.agent.associative_memory.active_conversation_with(
            agent_with.name)

        action_start_time = agent_with.scratch.time.time
        filling = []

        if conversation:
            filling = conversation.filling
            action_start_time = conversation.created

        filling += [ConversationFilling(name=self.agent.name, utterance=utterance, end=end)]
        description = await self.agent._generate_conversation_summary(filling)

        await self.agent._create_react_action(inserted_action=description,
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

        await agent_with._create_react_action(inserted_action=description,
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
            duration_minutes = round((self.agent.scratch.time.time - self.agent.scratch.action.start_time).total_seconds() / 60)
            await self.agent._update_schedule(description, duration_minutes)
            self.agent.scratch.chatting_with = None
            self.agent.scratch.chat = None
            self.agent.scratch.chatting_end_time = self.agent.scratch.time

    async def _wait_react(self, wait_time):
        event_short_description = self.agent.scratch.action.event.description.split(
            "(")[-1][:-1]

        inserted_action = f'waiting to start {event_short_description}'
        end_time = datetime.strptime(wait_time, "%B %d, %Y, %H:%M:%S")
        inserted_action_duration = round((end_time.minute + end_time.hour * 60) - (
            self.agent.scratch.time.time.minute + self.agent.scratch.time.time.hour * 60) + 1)

        action_address = f"<waiting> {self.agent.scratch.tile.x} {self.agent.scratch.tile.y}"
        action_event = (self.agent.name, "waiting to start", event_short_description)
        chatting_with = None
        chat = None
        chatting_with_buffer = None
        chatting_end_time = None

        action_pronunciatio = "⌛"

        await self.agent._update_schedule(inserted_action, inserted_action_duration)
        await self.agent._create_react_action(inserted_action, inserted_action_duration,
                                  action_address, action_event, chatting_with, chat, chatting_with_buffer, chatting_end_time,
                                  action_pronunciatio)


