"""

"""


import datetime
from enum import Enum
from typing import List
import uuid

from utils import get_date_string, get_time_string
from llm import llm
from memory import MemoryManager, MemoryType


class AgentState(Enum):
    IDLE = 0

class Agent:
    """ The agent is the main character in the simulation. Every round the agent is observing his environment. 
    He is always within a certain location and the location is keeping track of objects in the surrounding.
    """

    def __init__(self, name, initial_identity: str, current_location: str, memory: MemoryManager):
        self.id = uuid.uuid4()
        self.name = name
        self.memory = memory
        self.state = AgentState.IDLE
        self.current_location = current_location

        print(f"Agent[{self.id}] {self.name} created.")

        for identity_paragraph in initial_identity.split("\n"):
            memory.add(identity_paragraph, importance=10)
        
        print(f"Loaded memory with identity")

        self.energy = 100

    @property
    def identity(self):
        return "\n".join(self.memory.retrieve(f"Who is {self.name}?"))
    
    @property
    def long_term_plan(self):
        return self.memory.retrieve_by(MemoryType.LONG_TERM_PLAN)
    
    @property
    def short_term_plan(self):
        return self.memory.retrieve_by(MemoryType.SHORT_TERM_PLAN)

    def update(self, time: datetime = None, perceptions: List[str] = None) -> None:
        """ This method is used to update the agent. It is called by the tick method. """
        self.memorize(perceptions)

        # at 8:00 am the agent should create a new short term plan

        if not self.long_term_plan or time.hour == 8:
            self.update_long_term_plan(time)
            self.update_short_term_plan(time)

        if AgentState.IDLE and self.energy <= 40:
            self.reflect()
            self.state = AgentState.IDLE
            self.energy = 100
        
        for perception in perceptions:
            relevant_memories = self.memory.retrieve(perception)
            self.energy -= 1
            self.act(perception, relevant_memories)
            self.energy -= 1

    def update_short_term_plan(self, time: datetime):
        """ This method is used to update the short term plan. 
            Short term plan contains the daily schedule. Question is, based on the long term plan and his perception, 
            create a plan on which the Agent wants to do each hour of the day.
        """
        
        recent_experiences = self.memory.retrieve_most_recent_and_important_memories(10)
        daily_plan = self.memory.retrieve_by(MemoryType.LONG_TERM_PLAN)

        short_term_plan = llm.short_term_plan(self.name, self.identity, daily_plan, recent_experiences, get_time_string(time), get_date_string(time))
        self.memorize(short_term_plan, MemoryType.SHORT_TERM_PLAN)

    def update_long_term_plan(self, time):
        """ This method is used to update the long term plan. """

        # What is relevant for the Agent? What is his identity? What does he want to achieve
        # Query the LLM to generate this long term plan

        recent_experiences = self.memory.retrieve_most_recent_and_important_memories(10)
        known_locations = self.memory.retrieve_by(MemoryType.LOCATION)

        long_term_plan = llm.long_term_plan(self.name, self.identity, known_locations, recent_experiences, get_date_string(time))
        self.memorize(long_term_plan, MemoryType.LONG_TERM_PLAN)

    def should_update_short_term_plan(self, time) -> bool:
        """ This method is used to check if the short term plan should be updated. """
        return llm.should_update_short_term_plan(self.name, 
                                                 self.identity, 
                                                 self.short_term_plan, 
                                                 self.memory.retrieve_most_recent_and_important_memories(10),
                                                 get_date_string(time))

    def should_update_long_term_plan(self, time) -> bool:
        """ This method is used to check if the long term plan should be updated. """
        return llm.should_update_long_term_plan(self.name, 
                                                self.identity, 
                                                self.long_term_plan, 
                                                self.memory.retrieve_most_recent_and_important_memories(10), 
                                                get_date_string(time))

    def memorize(self, information: List[str], type: MemoryType = MemoryType.OBSERVATION) -> None:
        """ Functions to memorize the agent's memory. Every item of the list will be rated by the LLM in terms of importance"""
        ratings = [llm.rate_information_importance(info) for info in information]
        self.memory.add(information, importance=ratings, memory_type=type)

    def act(self, perception, relevant_memories):
        """ This method is used to act on the environment. It is called by the agent. """
        print(f"{perception}.")

        for memory in relevant_memories:
            print(memory)

    def reflect(self):
        """ This method is used to reflect on the agent's memory. It is called by the agent. """
        for entry in self.memory.as_stream():
            print(entry)