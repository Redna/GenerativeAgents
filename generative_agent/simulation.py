
""" this simulation should be able to be observed by agents. This file is the game engine which counts the steps and updates
its subscribers. import the necessary libraries
"""
import numpy as np
import dataclasses
from datetime import datetime, timedelta
from enum import Enum, auto
import time
import random
from typing import List, Optional


class Simulation:

    def __init__(self, minutes_per_tick=5):
        self.time = datetime.now()
        self.registry = []
        self.tick_delta = timedelta(minutes=minutes_per_tick)
        self.minutes_per_tick = minutes_per_tick
        self.ticks = 0
    
    def run(self, steps):
        for _ in range(steps):
            self.__tick()
    
    def __tick(self):
        self.time = self.time + self.tick_delta
        print(f"[#{self.ticks}] Simulation time: {self.time}")
        
        for element in self.registry:
            element.update(self.time)

        self.ticks += 1

    def spawn_agent(self, name, description, location):
        memory = MemoryManager()
        agent = Agent(name, description, location, memory)
        self.registry.append(agent)
        self.registry.append(memory)
        location
        return agent
    
    def spawn_object(self, name, description, location):
        object = Object(name, description)
        self.registry.append(object)
        return object
    
    def spawn_location(self, name, description, location):
        location = Location(name, description, location)
        self.registry.append(location)
        return location

class Object:
    """ An object can be anything that is visible in the environment. It can be a person, a tree, a rock, etc. """

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.state = f"{self.name} is does nothing"
    
    def update(self, time):
        self.steps += 1

    @property
    def state(self):
        return self.state
    
    def as_natural_language(self):
        return self.name + " " + self.description

class MemoryType(Enum):
    """ This enum is used to define the type of memory. """
    OBSERVATION = auto()
    CONVERSATION = auto()

    def __str__(self):
        return self.name.capitalize()
    
@dataclasses
class MemoryEntry:
    recency: float = 1.0
    importance: float = 0.5
    data: str
    embedding: Optional[any]
    type: MemoryType = MemoryType.OBSERVATION

class MemoryManager:
    def __init__(self, top_k = 10):
        self.entries = []
        self.top_k = top_k

    def add(self, data: str):
        self.entries.append(MemoryEntry(data=data))
    
    def retrieve(self, context: str):
        self._get_embedding(context)
        return self._filter(context)
    
    def _get_embedding(self, context: str):
        return ""

    def _filter(self, embedding: any):
        """ This method is used to find the most similar entries to the query. """

        relevent_memories = self.entries.sorted(key=lambda entry: self.score(embedding, entry), reverse=True)
        relevent_memories = relevent_memories[:self.top_k]

        for entry in relevent_memories:
            entry.recency = 1.0
        
        return relevent_memories

    def score(self, embedding: any, entry: MemoryEntry):
        """ This method is used to calculate the score of the entry. """
        relevance = np.dot(embedding, entry.embedding)
        return relevance * self.recency * self.importance

    def update():
        for entry in MemoryEntry:
            entry.recency = 0.95 * entry.recency

class Agent(Object):
    """ The agent is the main character in the simulation. Every round the agent is observing his environment. 
    He is always within a certain location and the location is keeping track of objects in the surrounding.
    """

    def __init__(self, name, location: 'Location', memory):
        self.energy = 100
        self.name = name
        self.position = 0
        self.steps = 0
        self.location = location
        self.is_reflecting = False

    def update(self, time: datetime = None):
        """ This method is used to update the agent. It is called by the tick method. """
        
        if self.energy <= 0 or self.is_reflecting:
            self.reflect()

        perceptions = self.perceive()
        self.energy -= 1

        context = self.memory.retrieve(perceptions)
        self.energy -= 1

        self.act(context)
        self.energy -= 1

        self.steps += 1

    def act(self):
        """ This method is used to interact with the environment. It is called by the agent. """
        pass

    def perceive(self):
        """ This method is used to observe the environment. It is called by the agent. """
        perceptions = []

        for oberservation in self.location.observe(self):
            perceptions.append(oberservation.as_natural_language())
            self.memory.add(oberservation)

        return perceptions

    def reflect(self):
        """ This method is used to reflect on the agent's memory. It is called by the agent. """
        print(self.memory_stream)

class Location:
    """ The location is the environment in which the agent is in. It keeps track of objects in the surrounding.
    """
    
    def __init__(self, name, description, neighbors: List['Location'] = None, objects: List[object] = None):
        self.objects = objects
        self.name = name
        self.description = description
        self.neighbors: List['Location'] = neighbors
    
    def add(self, object):
        self.objects.append(object)
    
    def observe(self, agent: 'Agent'):
        """ This method is used to observe the environment. It returns a list of objects in the surrounding. """
        
        observation = []

        for object in self.objects:
            if agent == object:
                continue
            observation.append(object.state)
            
        for neighbor in self.neighbors:
            observation.append(neighbor)


def main():
    simulation = Simulation()

    jacks_house = simulation.spawn_location("jacks_house", "A house")
    johns_house = simulation.spawn_location("johns_house", "A house where John lives")
    jacks_cafe = simulation.spawn_location("jacks_cafe", "A cafe where Jack drinks")
    wonder_work = simulation.spawn_location("wonder_work", "A wonderful workplace")

    simulation.spawn_agent("Jack", "A person", jacks_house)
    simulation.spawn_agent("John", "A person", johns_house)
    
    simulation.run(rounds=100)

if __name__ == "__main__":
    main()
