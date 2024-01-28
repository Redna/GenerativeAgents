import asyncio
from dataclasses import asdict
import datetime
from enum import Enum
import math
from operator import itemgetter
import random
from typing import Dict, List, Tuple
from copy import deepcopy

from async_lru import alru_cache
from generative_agents.communication.models import AgentDTO, MovementDTO
from generative_agents.conversational.chain.action_event_triple import ActionEventTriple
from generative_agents.conversational.chain.action_location_arena import ActionArenaLocations
from generative_agents.conversational.chain.action_location_game_object import ActionLocationGameObject
from generative_agents.conversational.chain.action_location_sector import ActionSectorLocations
from generative_agents.conversational.chain.action_pronunciatio import ActionPronunciatio
from generative_agents.conversational.chain.conversation import Conversation
from generative_agents.conversational.chain.conversation_summary import ConversationSummary
from generative_agents.conversational.chain.decide_to_react import DecideToReact
from generative_agents.conversational.chain.decide_to_talk import DecideToTalk
from generative_agents.conversational.chain.first_daily_plan import FirstDailyPlan
from generative_agents.conversational.chain.focused_event_to_context import FocusedEventToContext
from generative_agents.conversational.chain.new_decomposition_schedule import NewDecompositionSchedule
from generative_agents.conversational.chain.object_event import ObjectActionDescription
from generative_agents.conversational.chain.summarize_chat_relationship import ChatRelationshipSummarization
from generative_agents.conversational.chain.task_decomposition import TaskDecomposition
from generative_agents.conversational.chain.poignance import Poingnance
from generative_agents.conversational.chain.wake_up_hour import WakeUpHour
from generative_agents.core.memory.associative import AssociativeMemory
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.core.memory.scratch import Scratch
from generative_agents.core.whisper.whisper import whisper
from generative_agents.simulation.maze import Level, Maze, Tile
from generative_agents.conversational.chain import (daily_plan_and_status_chain,
                                                    reflection_points_chain,
                                                    evidence_and_insights_chain,
                                                    planning_on_conversation_chain,
                                                    memo_on_conversation_chain)
from generative_agents.conversational.chain.hourly_breakdown import HourlyBreakdown
from generative_agents.persistence import database
from generative_agents.persistence.database import ConversationFilling, initialize_agent
from generative_agents.simulation.time import SimulationTime, DayType
from generative_agents.core.events import Action, Event, ObjectAction, EventType, PerceivedEvent
from generative_agents.utils import get_time_string, hour_string_to_time


class ReactionMode(Enum):
    CHAT = "chat"
    WAIT = "wait"
    DO_OTHER_THINGS = "do other things"


class Agent:
    def __init__(self, name: str, age: int, description: str, innate_traits: List[str], time: SimulationTime, location: str, emoji: str, activity: str, tile: Tile, tree: MemoryTree = None):
        initialize_agent(name)
        self.name = name
        self.location = location
        self.emoji = emoji
        self.activity = activity
        self.scratch = Scratch(name=name, tile=tile, home=tile,
                               innate_traits=innate_traits, age=age)
        self.spatial_memory = MemoryTree() if not tree else tree
        self.associative_memory = AssociativeMemory(
            self.name, self.scratch.retention)
        self.time = time
        self.scratch.tile = tile
        self.scratch.description = description

        whisper(self.name, f"Initialized {self.name} at {self.scratch.tile}")

    def to_dto(self):
        return AgentDTO(
            name=self.name,
            age=self.scratch.age,
            inniate_traits=self.scratch.innate_traits,
            description=self.description,
            location=self.location,
            emoji=self.emoji,
            activity=self.activity,
            movement=MovementDTO(col=self.scratch.tile.x,
                                 row=self.scratch.tile.y)
        )

    @staticmethod
    def from_dto(dto: AgentDTO, maze: Maze, time: SimulationTime):
        return Agent(name=dto.name,
                     age=dto.age,
                     description=dto.description,
                     location=dto.location,
                     emoji=dto.emoji,
                     innate_traits=dto.inniate_traits,
                     activity=dto.activity,
                     time=time,
                     tile=maze.get_tile(dto.movement.col, dto.movement.row))

    @property
    def observation(self):
        if not self.scratch.action:
            return f"{self.name} is idle"
        else:
            event_description = self.scratch.action.event.description
            if "(" in event_description:
                event_description = event_description.split("(")[-1][:-1]

            if len(self.scratch.planned_path) == 0 and "waiting" not in event_description:
                return f"{self.name} is already {event_description}"

            if "waiting" in event_description:
                return f"{self.name} is {event_description}"

        return f"{self.name} is on the way to {event_description}"

    async def update(self, time: SimulationTime, maze: Maze, agents: Dict[str, 'Agent']) -> Tuple[Tile, str, str]:
        daytype: DayType = DayType.SAME_DAY

        if not self.scratch.time:
            daytype = daytype.FIRST_DAY
        elif (self.scratch.time.today != time.today):
            daytype = daytype.NEW_DAY

        self.scratch.time = time

        whisper(self.name, "is updating")
        perceived = await self.perceive(maze)
        whisper(self.name, f"perceived {len(perceived)} events")
        retrieved = await self.retrieve(perceived)
        whisper(self.name, f"retrieved {len(retrieved)} events")
        plan = await self.plan(agents, daytype, retrieved)
        whisper(self.name, f"planning to go to {plan}")
        await self.reflect()
        execution = await self.execute(maze, agents, plan)
        whisper(self.name, f"executing to {execution}")
        return execution

    async def perceive(self, maze: Maze):
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
                perception_tasks.append(self._perceive_event(self.scratch.action.event, type_=EventType.CHAT))
            else:
                perception_tasks.append(self._perceive_event(event))

        final_events = await asyncio.gather(*perception_tasks)

        for event in final_events:
            self.scratch.reflection_trigger_max -= event.poignancy * 10

        return final_events

    async def _perceive_event(self, event: Event, type_: EventType = EventType.EVENT):
        if type(event) != PerceivedEvent:
            event_poignancy = await self._rate_perception_poignancy(type_, event.description)

            whisper(self.name, f"event poignancy is {event_poignancy}")
            event = PerceivedEvent(**asdict(event), event_type=type_, poignancy=event_poignancy)
            event = self.associative_memory.add(event)
        return event


    async def retrieve(self, perceived: List[PerceivedEvent]) -> Dict[str, Dict[str, List[PerceivedEvent]]]:
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

    async def plan(self, agents: Dict[str, 'Agent'], daytype: DayType, retrieved: Dict[str, Dict[str, List[PerceivedEvent]]]):
        if daytype == DayType.NEW_DAY or daytype == DayType.FIRST_DAY:
            whisper(self.name, f"planning first daily plan")
            await self._long_term_planning(daytype)
        

        if self.scratch.is_action_finished():
            await self._determine_action()
            whisper(
                self.name, f"planning to {self.scratch.action.event.description}")

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
                    self.name, f"focusing on {focused_event['curr_event'].description}")
            else:
                whisper(self.name, f"not focusing on any persona event")

        # Step 2: Once we choose an event, we need to determine whether the
        #         persona will take any actions for the perceived event. There are
        #         three possible modes of reaction returned by _should_react.
        #         a) "chat with {target_persona.name}"
        #         b) "react"
        #         c) False

        if self.scratch.action.event.predicate == "chat with":
            last_utterance = self.scratch.action.event.filling[-1]
            if not last_utterance.end and last_utterance.name != self.name:
                await self._chat_react(agent_with=agents[last_utterance.name])
                focused_event = None

        if focused_event:
            reaction_mode, payload = await self._should_react(focused_event, agents)
            whisper(
                self.name, f"reaction mode is {reaction_mode} with payload {payload}")
            if reaction_mode and reaction_mode != ReactionMode.DO_OTHER_THINGS:
                # If we do want to chat, then we generate conversation
                if reaction_mode == ReactionMode.CHAT:
                    await self._chat_react(agent_with=payload)
                elif reaction_mode == ReactionMode.WAIT:
                    await self._wait_react(payload)
                # elif reaction_mode == "do other things":
                #   _chat_react(persona, focused_event, reaction_mode, personas)

        # Step 3: Chat-related state clean up.
        # If the persona is not chatting with anyone, we clean up any of the
        # chat-related states here.

        if self.scratch.action.event.predicate != "chat with":
            self.scratch.chatting_with = None
            self.scratch.chat = None
            self.scratch.chatting_end_time = None

        # We want to make sure that the persona does not keep conversing with each
        # other in an infinite loop. So, chatting_with_buffer maintains a form of
        # buffer that makes the persona wait from talking to the same target
        # immediately after chatting once. We keep track of the buffer value here.
        curr_persona_chat_buffer = self.scratch.chatting_with_buffer
        for persona_name, _ in curr_persona_chat_buffer.items():
            if persona_name != self.scratch.chatting_with:
                self.scratch.chatting_with_buffer[persona_name] -= 1

        return self.scratch.action.address

    async def reflect(self):
        if self.scratch.should_reflect():
            await self._run_reflect()
            whisper(self.name, f"reflected")
            self.scratch.reset_reflection_counter()

        last_conversation = self.associative_memory.last_conversation_with(
            self.scratch.chatting_with)

        # TODO check if this needs to be blocked
        if last_conversation and last_conversation.filling and last_conversation.filling[-1].end:
            evidence = [last_conversation.id]

            planning_thought = self._generate_planning_thought_on_conversation(
                last_conversation.filling)
            whisper(self.name, f"planning thought is {planning_thought}")
            planning_thought = f"For {self.scratch.name}'s planning: {planning_thought}"
            await self._add_reflection_thought(planning_thought, evidence)
            whisper(self.name, f"added reflection thought")

            memo_thought = await self._generate_memo_on_conversation(
                last_conversation.filling)
            memo_thought = f"{self.name} {memo_thought}"
            whisper(self.name, f"memo thought is {memo_thought}")
            await self._add_reflection_thought(memo_thought, evidence)
        pass

    async def execute(self, maze: Maze, agents: Dict[str, 'Agent'], plan):
        """
        Given a plan (action's string address), we execute the plan (actually 
        outputs the tile coordinate path and the next coordinate for the 
        persona). 

        INPUT:
            persona: Current <Persona> instance.  
            maze: An instance of current <Maze>.
            personas: A dictionary of all personas in the world. 
            plan: This is a string address of the action we need to execute. 
            It comes in the form of "{world}:{sector}:{arena}:{game_objects}". 
            It is important that you access this without doing negative 
            indexing (e.g., [-1]) because the latter address elements may not be 
            present in some cases. 
            e.g., "dolores double studio:double studio:bedroom 1:bed"

        OUTPUT: 
            execution
        """
        if "<random>" in plan or self.scratch.planned_path == []:
            self.scratch.action_path_set = False

        # <action_path_set> is set to True if the path is set for the current action.
        # It is False otherwise, and means we need to construct a new path.
        if not self.scratch.action_path_set:
            # <target_tiles> is a list of tile coordinates where the persona may go
            # to execute the current action. The goal is to pick one of them.
            target_tiles = None

            if "<persona>" in plan:
                # Executing persona-persona interaction.
                target_persona_tile = agents[plan.split(
                    "<persona>")[-1].strip()].scratch.tile
                potential_path = maze.find_path(self.scratch.tile,
                                                target_persona_tile)

                if len(potential_path) <= 2:
                    target_tiles = [potential_path[0]]
                else:
                    potential_1 = maze.find_path(self.scratch.tile,
                                                 potential_path[int(len(potential_path)/2)])
                    potential_2 = maze.find_path(self.scratch.tile,
                                                 potential_path[int(len(potential_path)/2)+1])
                    if len(potential_1) <= len(potential_2):
                        target_tiles = [
                            potential_path[int(len(potential_path)/2)]]
                    else:
                        target_tiles = [
                            potential_path[int(len(potential_path)/2+1)]]

            elif "<waiting>" in plan:
                # Executing interaction where the persona has decided to wait before
                # executing their action.
                x = int(plan.split()[1])
                y = int(plan.split()[2])
                target_tiles = [[x, y]]

            elif "<random>" in plan:
                # Executing a random location action.
                target_tiles = [maze.get_random_tile(self.scratch.tile)]
            else:
                # This is our default execution. We simply take the persona to the
                # location where the current action is taking place.
                # Retrieve the target addresses. Again, plan is an action address in its
                # string form. <maze.address_tiles> takes this and returns candidate
                # coordinates.
                if plan not in maze.address_tiles:
                    fallback_plan = ":".join(plan.split(":")[0:-1])

                    if fallback_plan not in maze.address_tiles:
                        fallback_plan = random.choice(
                            list(maze.address_tiles.keys()))

                    target_tiles = maze.address_tiles[fallback_plan]
                else:
                    target_tiles = maze.address_tiles[plan]

            # There are sometimes more than one tile returned from this (e.g., a tabe
            # may stretch many coordinates). So, we sample a few here. And from that
            # random sample, we will take the closest ones.
            if len(target_tiles) < 4:
                target_tiles = random.sample(
                    list(target_tiles), len(target_tiles))
            else:
                target_tiles = random.sample(list(target_tiles), 4)

            # If possible, we want personas to occupy different tiles when they are
            # headed to the same location on the maze. It is ok if they end up on the
            # same time, but we try to lower that probability.
            # We take care of that overlap here.
            persona_name_set = set(agents.keys())
            new_target_tiles = []
            for tile in target_tiles:
                curr_event_set = tile.events
                pass_curr_tile = False
                for j in curr_event_set:
                    if j[0] in persona_name_set:
                        pass_curr_tile = True
                if not pass_curr_tile:
                    new_target_tiles += [tile]
            if len(new_target_tiles) == 0:
                new_target_tiles = target_tiles
            target_tiles = new_target_tiles
            # Now that we've identified the target tile, we find the shortest path to
            # one of the target tiles.
            curr_tile = self.scratch.tile
            closest_target_tile = None
            path = None
            for i in target_tiles:
                # path_finder takes a collision_mze and the curr_tile coordinate as
                # an input, and returns a list of coordinate tuples that becomes the
                # path.
                # e.g., [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4)...]
                curr_path = maze.find_path(curr_tile, i)

                if not closest_target_tile:
                    closest_target_tile = i
                    path = curr_path
                elif len(curr_path) < len(path):
                    closest_target_tile = i
                    path = curr_path

            # Actually setting the <planned_path> and <action_path_set>. We cut the
            # first element in the planned_path because it includes the curr_tile.
            self.scratch.planned_path = path[1:]
            self.scratch.action_path_set = True

        # Setting up the next immediate step. We stay at our curr_tile if there is
        # no <planned_path> left, but otherwise, we go to the next tile in the path.
        ret = self.scratch.tile
        if self.scratch.planned_path:
            ret = self.scratch.planned_path[0]
            self.scratch.planned_path = self.scratch.planned_path[1:]

        description = f"{self.scratch.action.event.description}"
        description += f" @ {self.scratch.action.address}"

        self.emoji = self.scratch.action.emoji
        self.description = description
        return ret

    @alru_cache(maxsize=512)
    async def _rate_perception_poignancy(self, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = await Poingnance(agent_name=self.name,
                                 agent_identity=await self.scratch.identity,
                                 description=description,
                                 type_=event_type.value).run()

        # TODO properly check the output here
        return int(score) / 10

    async def _add_reflection_thought(self, thought: str, evidence: List[str]):
        created = self.scratch.time
        expiration = created + datetime.timedelta(days=30)
        s, p, o = ActionEventTriple(self.name, thought)

        thought_poignancy = await self._rate_perception_poignancy(EventType.THOUGHT, thought)

        perceived_event = PerceivedEvent(subject=s, predicate=p, object_=o, description=thought,
                                         event_type=EventType.THOUGHT, poignancy=thought_poignancy,
                                         filling=evidence, expiration=expiration, created=created)
        self.associative_memory.add(perceived_event)

    @staticmethod
    def __utterances_to_conversation(utterances: List[ConversationFilling]):
        return "\n".join([f"{i.name}: {i.utterance}" for i in utterances])

    async def _generate_memo_on_conversation(self, utterances: List[ConversationFilling]):
        return await memo_on_conversation_chain.arun(conversation=self.__utterances_to_conversation(utterances),
                                                     agent=self.name)

    async def _generate_planning_thought_on_conversation(self, utterances: List[ConversationFilling]):
        return await planning_on_conversation_chain.arun(conversation=self.__utterances_to_conversation(utterances),
                                                         agent=self.name)

    async def _run_reflect(self):
        """
        Run the actual reflection. We generate the focal points, retrieve any 
        relevant nodes, and generate thoughts and insights. 

        INPUT: 
            persona: Current Persona object
        Output: 
            None
        """
        # Reflection requires certain focal points. Generate that first.
        focal_points = await self._generate_reflection_points(3)
        whisper(self.name, f"generated {focal_points} focal points")
        # Retrieve the relevant Nodes object for each of the focal points.
        # <retrieved> has keys of focal points, and values of the associated Nodes.
        retrieved = await self.associative_memory.retrieve_relevant_entries(
            focal_points)

        whisper(self.name, f"retrieved {len(retrieved)} relevant nodes")

        # For each of the focal points, generate thoughts and save it in the
        # agent's memory.
        for nodes in retrieved:
            thoughts = await self._generate_insights_and_evidence(nodes, 5)
            for thought, evidence in thoughts.items():
                await self._add_reflection_thought(thought, evidence)

    async def _generate_insights_and_evidence(self, memories: List[PerceivedEvent], num_insights: int):
        """
        Generate insights and evidence for the given memories. 
        INPUT: 
            memories: A list of <PerceivedEvent> instances that are the memories 
                that we want to generate insights for. 
            num_insights: The number of insights that we want to generate. 
        OUTPUT: 
            insights: A dictionary that contains the generated insights. 
                insights[insight] = evidence
        """

        # TODO parse the output properly
        statements = '\n'.join(
            [f'{str(count)}. {node.embedding_key}' for count, node in enumerate(memories, 1)])

        return await evidence_and_insights_chain.arun(statements=statements,
                                                      num_insights=num_insights)

    async def _generate_reflection_points(self, num_points: int):
        memories = self.associative_memory.get_most_recent_memories(num_points)
        return await reflection_points_chain.arun(memories=memories,
                                                  count=num_points)

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

            if await self._generate_decide_to_talk(target_agent, retrieved):
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

            react_mode = await self._generate_decide_to_react(target_agent, retrieved)

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
        if self.scratch.chatting_with:
            return ReactionMode.DO_OTHER_THINGS, None
        if "<waiting>" in self.scratch.action.address:
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

            if (target_agent.name in self.scratch.chatting_with_buffer):
                if self.scratch.chatting_with_buffer[target_agent.name] > 0:
                    return ReactionMode.DO_OTHER_THINGS, None

            if await lets_talk(self, agents[curr_event.subject], focused_event):
                return ReactionMode.CHAT, target_agent

            return await lets_react(self, agents[curr_event.subject], focused_event)

        return ReactionMode.DO_OTHER_THINGS, None

    async def _chat_react(self, agent_with: 'Agent'):
        utterance, end = await self._generate_conversation(agent_with)
        conversation = self.associative_memory.active_conversation_with(
            agent_with.name)

        action_start_time = agent_with.scratch.time.time
        filling = []

        if conversation:
            filling = conversation.filling
            action_start_time = conversation.created

        filling += [ConversationFilling(name=self.name, utterance=utterance, end=end)]
        description = await self._generate_conversation_summary(filling)

        await self._create_react_action(inserted_action=description,
                                                     inserted_action_duration=10,
                                                     action_address=f"<persona> {agent_with.name}",
                                                     action_event=(
                                                         self.name, "chat with", agent_with.name),
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
                                        action_address=f"<persona> {self.name}",
                                        action_event=(
                                            agent_with.name, "chat with", self.name),
                                        chatting_with=agent_with.name,
                                        chat="-",
                                        chatting_with_buffer={
                                            agent_with.name: 800},
                                        chatting_end_time=None,
                                        action_pronunciatio="💬",
                                        filling=filling,
                                        action_start_time=action_start_time)

        if end:
            duration_minutes = round((self.scratch.time.time - self.scratch.action.start_time).total_seconds() / 60)
            await self._update_schedule(description, duration_minutes)
            self.scratch.chatting_with = None
            self.scratch.chat = None
            self.scratch.chatting_end_time = self.scratch.time

    async def _generate_conversation(self, agent_with: 'Agent'):
        retrieved = self.associative_memory.retrieve_relevant_entries(
            [agent_with.name], 50)
        relationship = await self._generate_summarize_agent_relationship(
            agent_with, retrieved)

        focal_points = [
            f"{relationship}", f"{agent_with.name} is {agent_with.scratch.action.event.description}"]
        active_conversation = self.associative_memory.active_conversation_with(
            agent_with.name)

        active_conversation_string = ""
        if active_conversation:
            active_conversation_string = "\n".join(
                [f"{filling.name}: {filling.utterance}" for filling in active_conversation.filling])
            focal_points += [active_conversation_string]
        else:
            active_conversation = ""

        retrieved += self.associative_memory.retrieve_relevant_entries(
            focal_points, 15)
        location = f"{self.scratch.tile.arena} in {self.scratch.tile.sector}"
        description = list(set([event.description for event in retrieved]))
        memory = "\n".join(description)

        last_conversation = self.associative_memory.last_conversation_with(
            agent_with.name)

        past_context = ""

        if last_conversation and (previous_chat := int((self.scratch.time.time - last_conversation.created).total_seconds()/60)) > 480:
            past_context = f"{str(previous_chat)} minutes ago, {self.name} and {agent_with.name} were already {last_conversation.description}. This context takes place after that conversation."

        utterance, end = await Conversation(identity=await self.scratch.identity,
                                            agent=self.name,
                                            agent_with=agent_with.name,
                                            agent_with_action=agent_with.scratch.action.event.description,
                                            agent_action=self.scratch.action.event.description,
                                            location=location,
                                            conversation=active_conversation_string,
                                            memory=memory,
                                            past_context=past_context).run()

        return utterance, end

    async def _generate_conversation_summary(self, conversation_filling: List[ConversationFilling]):
        conversation_history = "\n".join(
            [f"{filling.name}: {filling.utterance}" for filling in conversation_filling])
        return await ConversationSummary(agent=self.name, conversation=conversation_history).run()

    async def _generate_summarize_agent_relationship(self, agent_with: 'Agent', retrieved: List[PerceivedEvent]):
        """
        We generate the relationship between the two agents. 
        """
        statements = "\n".join([event.description for event in retrieved])
        relationship = await ChatRelationshipSummarization(agent=self.name,
                                                           agent_with=agent_with.name,
                                                           statements=statements).run()
        return relationship

    async def _wait_react(self, wait_time):
        event_short_description = self.scratch.action.event.description.split(
            "(")[-1][:-1]

        inserted_action = f'waiting to start {event_short_description}'
        end_time = datetime.strptime(wait_time, "%B %d, %Y, %H:%M:%S")
        inserted_action_duration = round((end_time.minute + end_time.hour * 60) - (
            self.scratch.time.time.minute + self.scratch.time.time.hour * 60) + 1)

        action_address = f"<waiting> {self.scratch.tile.x} {self.scratch.tile.y}"
        action_event = (self.name, "waiting to start", event_short_description)
        chatting_with = None
        chat = None
        chatting_with_buffer = None
        chatting_end_time = None

        action_pronunciatio = "⌛"

        await self._update_schedule(inserted_action, inserted_action_duration)
        await self._create_react_action(inserted_action, inserted_action_duration,
                                  action_address, action_event, chatting_with, chat, chatting_with_buffer, chatting_end_time,
                                  action_pronunciatio)

    def __calculate_start_end_hours(self):
        min_sum = sum(duration for _, duration in self.scratch.daily_schedule_hourly_organzied[:self.scratch.get_daily_schedule_index()])
        start_hour = int(min_sum / 60)

        current_activity, next_activity = self.scratch.daily_schedule_hourly_organzied[self.scratch.get_daily_schedule_index():self.scratch.get_daily_schedule_index()+2]

        if current_activity[1] >= 120:
            end_hour = start_hour + current_activity[1] / 60
        elif current_activity[1] + next_activity[1] >= 120:
            end_hour = start_hour + (current_activity[1] + next_activity[1]) / 60
        else:
            end_hour = start_hour + 2

        return int(start_hour), int(end_hour)

    def __get_start_and_end_index(self, start_hour, end_hour):
        duration_sum = 0
        start_index = end_index = None

        for index, (_, duration) in enumerate(self.scratch.daily_schedule):
            if duration_sum >= start_hour * 60 and start_index is None:
                start_index = index
            if duration_sum >= end_hour * 60 and end_index is None:
                end_index = index
            duration_sum += duration
        
        return start_index, end_index

    async def _update_schedule(self, inserted_action, inserted_action_duration):
        start_hour, end_hour = self.__calculate_start_end_hours()
        start_index, end_index = self.__get_start_and_end_index(start_hour, end_hour)
        
        """
        agent: str
        start_hour: str
        end_hour: str
        new_event: str
        new_event_duration: str
        new_event_index: int
        schedule_slice: List[Dict[str, str]]
        """


        new_schedule = await NewDecompositionSchedule(agent=self.name,
                                                        start_hour=start_hour,
                                                        end_hour=end_hour,
                                                        new_event=inserted_action,
                                                        new_event_duration=inserted_action_duration,
                                                        new_event_index=self.scratch.get_daily_schedule_index() - start_index,
                                                        schedule_slice=self.scratch.daily_schedule_hourly_organzied[start_index:end_index]).run()
        
        self.scratch.daily_schedule[start_index:end_index] = new_schedule

    async def _create_react_action(self, inserted_action, inserted_action_duration,
                             action_address, action_event, chatting_with, chat, chatting_with_buffer,
                             chatting_end_time, action_pronunciatio, filling=[], action_start_time=None):

        event_poignancy = await self._rate_perception_poignancy(EventType.CHAT, inserted_action)
        
        event = PerceivedEvent(depth=0,
                      subject=self.name,
                      predicate=action_event[1],
                      object_=action_event[2],
                      description=inserted_action,
                      filling=filling,
                      poignancy=event_poignancy,
                      event_type=EventType.CHAT,
                      tile=self.scratch.tile)
        
        next_action = Action(address=action_address,
                             start_time=action_start_time,
                             duration=inserted_action_duration,
                             emoji=action_pronunciatio,
                             event=event)

        self.scratch.finished_action_queue.put(self.scratch.action)
        self.scratch.action = next_action
        self.scratch.chatting_with = chatting_with
        self.scratch.chat = chat
        if chatting_with_buffer:
            self.scratch.chatting_with_buffer = {**self.scratch.chatting_with_buffer, **chatting_with_buffer}
        self.scratch.chatting_end_time = chatting_end_time
        self.associative_memory.add(event)

    @staticmethod
    def _focused_event_to_context(retrieved: Dict[str, List[PerceivedEvent]]):
        context = ""
        # TODO think about a better way to convert to past tense
        for event in retrieved["events"]:
            description = event.description.split(" ")
            description[2:3] = ["was"]
            description = " ".join(description)
            context += f"{description}. "
        context += "\n"
        for thought in retrieved["thoughts"]:
            context += f"{thought.description}. "

        return context

    async def _generate_decide_to_talk(self, target_agent: 'Agent', retrieved: Dict[str, List[PerceivedEvent]]):
        context = await FocusedEventToContext(identity=await self.scratch.identity,
                                              agent=self.name,
                                              event_description=retrieved["curr_event"].description,
                                              events=", ".join(
                                                  [event.description for event in retrieved["events"]]),
                                              thoughts=", ".join([thought.description for thought in retrieved["thoughts"]])).run()

        last_conversation = self.associative_memory.last_conversation_with(
            target_agent.name)

        if last_conversation:
            last_chat_summary = f"last chatted at {last_conversation.created.strftime('%B %d, %Y, %H:%M:%S')} about {last_conversation.description}"
        else:
            last_chat_summary = "never chatted before."

        return await DecideToTalk(context=context,
                                  current_time=self.scratch.time.as_string(),
                                  init_agent=self.name,
                                  init_agent_observation=self.scratch.action.event.description,
                                  agent_with=target_agent.name,
                                  agent_with_observation=retrieved["curr_event"].description,
                                  last_chat_summary=last_chat_summary).run()

    async def _generate_decide_to_react(self, target_agent: 'Agent', retrieved: Dict[str, List[PerceivedEvent]]):
        context = self._focused_event_to_context(retrieved)
        current_time = self.scratch.time.as_string()

        return await DecideToReact(context=context,
                                   current_time=current_time,
                                   agent=self.name,
                                   agent_with=target_agent.name,
                                   agent_with_action=target_agent.scratch.action.event.description,
                                   agent_observation=self.observation,
                                   agent_with_observation=target_agent.observation,
                                   initial_action_description=self.scratch.action.event.description).run()

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
        no_self_event_retrieved = {description: context for description, context in retrieved.items(
        ) if context["curr_event"].subject != self.name}

        persona_context = [context for _, context in no_self_event_retrieved.items(
        ) if ":" not in context["curr_event"].subject]
        if persona_context:
            return random.choice(persona_context)

        non_idle_context = [context for _, context in no_self_event_retrieved.items(
        ) if "idle" not in context["curr_event"].description]
        if non_idle_context:
            return random.choice(non_idle_context)

        return None

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
                                                1] = await self._decompose_action(current_index, action_description, action_duration)

            if next_hour_index + 1 < len(self.scratch.daily_schedule):
                action_description, action_duration = self.scratch.daily_schedule[next_hour_index]

                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        self.scratch.daily_schedule[next_hour_index:next_hour_index + 1] = await self._decompose_action(
                            next_hour_index, action_description, action_duration)

        if next_hour_index < len(self.scratch.daily_schedule):
            # If it is not the first hour of the day, this is always invoked (it is
            # also invoked during the first hour of the day -- to double up so we can
            # decompose two hours in one go). Of course, we need to have something to
            # decompose as well, so we check for that too.
            if self.scratch.time.time.hour < 23:
                # And we don't want to decompose after 11 pm.
                action_description, action_duration = self.scratch.daily_schedule[next_hour_index]
                if action_duration >= 60:
                    if needs_decomposition(action_description, action_duration):
                        # current_index:next_hour_index
                        decomposition = await self._decompose_action(next_hour_index, action_description, action_duration)
                        self.scratch.daily_schedule[next_hour_index:next_hour_index+1] = (
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

        action_description, action_duration = self.scratch.daily_schedule[current_index]

        action_game_object = None
        next_address = ""
        try:
            whisper(
                self.name, f"determined next action: {action_description}")
            action_sector = await self._generate_next_action_sector(action_description)

            if action_sector == "<random>":
                next_address = action_sector
                tile = None
            else:
                whisper(self.name, f"determined next sector: {action_sector}")
                action_arena = await self._generate_next_action_arena(
                    action_description, action_sector)
                whisper(self.name, f"determined next arena: {action_arena}")
                next_address = await self._generate_next_action_game_object(
                    action_description, action_arena)
                
                address_parts = next_address.split(":")
                tile = self.spatial_memory[address_parts[0]][address_parts[1]][address_parts[2]].game_objects[address_parts[3]]

                whisper(self.name, f"determined next game object: {action_game_object}")
        except Exception as e:
            print(f"Unable to generate next action: {e}")
            next_address = "<random>"
            tile = None

        action_pronouncio = await self._generate_action_pronunciatio(
            action_description)
        whisper(self.name, f"determined next pronouncio: {action_pronouncio}")

        action_event = await self._generate_action_event_triple(action_description)
        whisper(self.name, f"determined next event triple: {action_event}")

        object_action = None
        if next_address != "<random>":
            action_object_desctiption, tripplet = await self._generate_action_object_description(
                next_address, action_description)
            whisper(self.name, f"determined next object description: {action_object_desctiption}")
            action_object_pronunciatio = await self._generate_action_pronunciatio(
                action_object_desctiption)
            whisper(self.name, f"determined next object pronouncio: {action_object_pronunciatio}")
            subject, predicate, object_ = tripplet
            whisper(self.name, f"determined next object event triple: {subject}, {predicate}, {object_}")

            object_action = ObjectAction(address=next_address,
                                     emoji=action_object_pronunciatio,
                                     event=Event(subject=subject,
                                                 predicate=predicate,
                                                 object_=object_,
                                                 description=action_object_desctiption,
                                                 depth=0,
                                                 tile=tile))

        minutes_from_now = self.time.time.hour * 60 + self.time.time.minute

        planned_end = 0
        for _, duration in self.scratch.daily_schedule[:current_index+1]:
            planned_end += int(duration)

        minutes_left = planned_end - minutes_from_now + 1

        next_action = Action(address=next_address,
                             start_time=self.scratch.time.time,
                             duration=minutes_left,
                             emoji=action_pronouncio,
                             event=Event(subject=self.name,
                                         predicate=action_event[1],
                                         object_=action_event[2],
                                         description=action_description,
                                         depth=0, 
                                         tile=tile
                                         ),
                             object_action=object_action)

        if self.scratch.action:
            self.scratch.finished_action_queue.put(self.scratch.action)
        self.scratch.action = next_action

    async def _generate_action_object_description(self, action_game_object, action_description):

        object_name = action_game_object.split(":")[-1]

        action_object_description = await ObjectActionDescription(name=self.name,
                                                                  object_name=object_name,
                                                                  object_address=action_game_object,
                                                                  action_description=action_description).run()
        return action_object_description

    async def _generate_action_event_triple(self, action_description):
        action_event = await ActionEventTriple(name=self.name,
                                               action_description=action_description).run()

        return action_event

    async def _generate_action_pronunciatio(self, action_description):
        action_pronouncio = await ActionPronunciatio(action_description=action_description).run()
        return action_pronouncio

    async def _generate_next_action_game_object(self, action_description, action_arena):
        arena = action_arena
        available_objects = self.spatial_memory.get_str_accessible_arena_game_objects(
            action_arena)

        if not available_objects:
            arena = self.scratch.tile.get_path(Level.ARENA)
            available_objects = self.spatial_memory.get_str_accessible_arena_game_objects(
                arena)

        retry = 0
        game_object = await ActionLocationGameObject(action_description=action_description,
                                                     available_objects=available_objects,
                                                     retry=str(retry)).run()

        return f"{arena}:{game_object}"

    async def _generate_next_action_arena(self, action_description, action_sector):
        name = self.name
        current_sector = self.scratch.tile.sector
        current_area = self.scratch.tile.arena
        sector = action_sector.split(":")[-1]

        arena = await ActionArenaLocations(name=name,
                                           current_area=current_area,
                                           current_sector=current_sector,
                                           sector=sector,
                                           sector_arenas=self.spatial_memory.get_str_accessible_sector_arenas(
                                               action_sector),
                                           available_sectors_nearby=self.spatial_memory.get_str_accessible_sectors(
                                               list(self.spatial_memory.tree.keys())[-1]),
                                           action_description=action_description).run()

        return f"{action_sector}:{arena}"

    async def _generate_next_action_sector(self, action_description):
        name = self.name
        home = self.scratch.home.sector
        home_arenas = self.spatial_memory.get_str_accessible_sector_arenas(
            self.scratch.home.get_path(Level.SECTOR))
        current_sector = self.scratch.tile.sector
        current_sector_arenas = self.spatial_memory.get_str_accessible_sector_arenas(
            self.scratch.tile.get_path(Level.SECTOR))
        nearby_sectors = self.spatial_memory.get_str_accessible_sectors(
            self.scratch.tile.get_path(Level.WORLD))

        next_sector = await ActionSectorLocations(agent_name=name,
                                                  agent_home=home,
                                                  agent_home_arenas=home_arenas,
                                                  agent_current_sector=current_sector,
                                                  agent_current_sector_arenas=current_sector_arenas,
                                                  available_sectors_nearby=nearby_sectors,
                                                  curr_action_description=action_description).run()

        return f"{self.scratch.tile.world}:{next_sector}" if next_sector != "<random>" else next_sector

    async def _decompose_action(self, action_index: int, action_description: str, action_duration: int):
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
        if current_index+2 < len(self.scratch.daily_schedule):
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

        task_context = f"From {', and '.join(task_sumaries)}."

        start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                      + datetime.timedelta(minutes=start_min)).strftime("%H:%M:%S")
        end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                    + datetime.timedelta(minutes=end_min)).strftime("%H:%M:%S")

        return await TaskDecomposition(name=self.name,
                                       identity=await self.scratch.identity,
                                       today=self.scratch.time.today,
                                       task_context=task_context,
                                       task_description=action_description,
                                       task_duration=str(action_duration),
                                       task_start_time=start_time,
                                       task_end_time=end_time).run()

    async def _long_term_planning(self, daytype: DayType):
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
        wake_up_hour = await WakeUpHour(agent_name=self.name,
                                        agent_lifestyle=self.scratch.lifestyle,
                                        agent_identity=await self.scratch.identity).run()

        whisper(self.name, f"wake up hour is at {wake_up_hour}")

        # When it is a new day, we start by creating the daily_req of the persona.
        # Note that the daily_req is a list of strings that describe the persona's
        # day in broad strokes.
        if daytype == DayType.FIRST_DAY:
            # Bootstrapping the daily plan for the start of then generation:
            # if this is the start of generation (so there is no previous day's
            # daily requirement, or if we are on a new day, we want to create a new
            # set of daily requirements.
            self.scratch.daily_requirements = await FirstDailyPlan(agent_name=self.name,
                                                                   agent_identity=await self.scratch.identity,
                                                                   agent_lifestyle=self.scratch.lifestyle,
                                                                   current_day=self.scratch.time.today,
                                                                   wake_up_hour=wake_up_hour).run()
            whisper(
                self.name, f"first daily plan is {self.scratch.daily_requirements}")
        elif daytype == DayType.NEW_DAY:
            # TODO parse daily_plan to a list
            daily_plan, current_status = self._generate_daily_plan_and_current_status()

            whisper(self.name, f"new daily plan is {daily_plan}")
            whisper(self.name, f"new current status is {current_status}")

            self.scratch.daily_requirements = daily_plan
            self.scratch.current_status = current_status

        # Based on the daily_req, we create an hourly schedule for the persona,
        # which is a list of todo items with a time duration (in minutes) that
        # add up to 24 hours.

        self.scratch.daily_schedule = await HourlyBreakdown(identity=await self.scratch.identity,
                                                            wake_up_hour=wake_up_hour,
                                                            name=self.name,
                                                            hourly_organized_activities=self.scratch.daily_requirements).run()
        
        daily_plan = ",".join([entry['activity'] for entry in self.scratch.daily_schedule])
        description = f"This is {self.name}'s plan for {self.scratch.time.today}: {daily_plan}."
        self.scratch.daily_schedule_hourly_organzied = self.scratch.daily_schedule = [(entry['activity'], 60) for entry in self.scratch.daily_schedule]

        whisper(self.name, f"new daily plan is {description}")

        perceived_plan = PerceivedEvent(event_type=EventType.PLAN,
                                        poignancy=0.5,
                                        depth=1,
                                        description=description,
                                        subject=self.name,
                                        predicate="plan",
                                        object_=self.scratch.time.today,
                                        created=self.scratch.time.time,
                                        expiration=self.scratch.time.time + datetime.timedelta(days=30),
                                        tile=self.scratch.tile)

        self.associative_memory.add(perceived_plan)

    async def _generate_daily_plan_and_current_status(self) -> Tuple[str, str]:

        retrieved_events = self._get_related_to_text(
            f"{self.name}'s plan for {self.scratch.time.as_string()}.", EventType.PLAN)
        retrieved_events += self._get_related_to_text(
            f"Important recent events for {self.name}'s life.")

        statements = "[Statements]\n"

        for retrieved_event in retrieved_events:
            statements += f"{retrieved_event.created.strftime('%A %B %d -- %H:%M %p')}: {retrieved_event.description}\n"
        # "name", "today", "yesterday", "statements", "today", "current_activity"

        daily_plan_and_status = await daily_plan_and_status_chain.arun(agent_name=self.name,
                                                                       today=self.scratch.time.today,
                                                                       yesterday=self.scratch.time.yesterday,
                                                                       statements=self.scratch.associative_memory.latest_events_summary,
                                                                       current_activity=self.scratch.current_activity)

        return daily_plan_and_status["daily_plan"], daily_plan_and_status["currently"]

    def _get_related_to_text(self, text: str,  event_type: EventType = None):
        if event_type:
            memories = database.get_by_type(self.name, text, event_type)
        else:
            memories = database.get(self.name, text)

        return [PerceivedEvent.from_db_entry(memory) for memory in memories]

    def _get_related_events(self, event: PerceivedEvent, event_type: EventType = None):
        return self._get_related_to_text(event.description, event_type)
