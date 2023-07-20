from datetime import datetime
from agent import Agent
from memory import MemoryManager


class Simulation:

    def __init__(self):
        self.registry = []
    
    def spawn_agent(self, name, description, location):
        memory = MemoryManager()
        agent = Agent(name, description, location, memory)
        self.registry.append(agent)
        self.registry.append(memory)
        location
        return agent

    def update(self):
        #TODO add some fancy logic to perceive things for the agent
        for object in self.registry:
            object.update()
       
simulation = Simulation()

if __name__ == "__main__":
    simulation.spawn_agent("Bob", "Bob is a nice guy", "Home")
    simulation.spawn_agent("Alice", "Alice is a nice girl", "Home")

    simulation.registry[0].update(datetime.now(), ["Alice is walking around", "The door to the house is closed", "The fridge is empty"])