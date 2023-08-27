from threading import Thread
import api
import heapq
import asyncio
from copy import deepcopy
import datetime
from typing import Coroutine, List
from api import AgentDTO, MovementDTO, RoundUpdateDTO
from maze import Maze, Tile


class Scratch():
    def __init__(self):
        self.planned_path: List[Tile] = []

class Agent():
    def __init__(self, name: str, description: str, location: str, emoji: str, activity: str, tile: Tile):
        self.name = name
        self.description = description
        self.location = location
        self.tile = tile
        self.emoji = emoji
        self.activity = activity
        self.scratch = Scratch()

    def to_dto(self):
        return AgentDTO(
            name=self.name,
            description=self.description,
            location=self.location,
            emoji=self.emoji,
            activity=self.activity,
            movement=MovementDTO(col=self.tile.x, row=self.tile.y)
        )
    
    @staticmethod
    def from_dto(dto: AgentDTO, maze: Maze):
        return Agent(name=dto.name,
                     description=dto.description,
                     location=dto.location,
                     emoji=dto.emoji,
                     activity=dto.activity,
                     tile=maze.get_tile(dto.movement.col, dto.movement.row))
    
    def update(self, time: datetime.datetime, maze: Maze):
        while not self.scratch.planned_path:
            print("Searching for a suitable path for the agent")
            target_tile = maze.get_random_tile(self.tile)
            path = maze.find_path(self.tile, target_tile)
            print("Setting new target to ", target_tile)
            self.scratch.planned_path = path
            if not path: 
                print("Here is no path to the target, trying again")

        self.tile = self.scratch.planned_path.pop(0)
        print("Moving to", self.tile)

class RoundUpdateSnapshots():
    def __init__(self):
        self.rounds = []

    def add(self, time, agents: List[Agent]):
        agents_dto = [agent.to_dto() for agent in agents.values()]
        
        converted_date_time = time.as_string()
        round_update = RoundUpdateDTO(round=len(self.rounds), time=converted_date_time, agents=agents_dto)
        self.rounds += [round_update]

    def get(self, round: int):
        return self.rounds[round]
    
    def get_all(self):
        return self.rounds
    
    @property
    def last(self):
        return self.rounds[-1]

    @property
    def current_round(self):
        return len(self.rounds)


class SimulationTime():
    """
    A class to represent the simulation time. It uses the the real time using datetime module but increments it every step with a 
    certain amount of time. 
    """

    def __init__(self, increment: int):
        self.increment = increment
        self.time = datetime.datetime.now()

    def tick(self):
        self.time += datetime.timedelta(seconds=self.increment)

    def get(self):
        return self.time
    
    def as_string(self):
        """
        Returns the time as a string

        e.g. 30th of August 2021, 12:00:00
        """
        return self.time.strftime("%d %B %Y, %H:%M:%S")

class Simulation():
    def __init__(self, round_updates: RoundUpdateSnapshots):
        self.maze = Maze()
        self.agents = dict()
        self.round_updates = round_updates
        self.simulated_time = SimulationTime(5)
    
    def spawn_agent(self, data: AgentDTO):
        print(f"spawning agent {data.name}, at {data.movement.col}, {data.movement.row}")
        self.agents[data.name] = Agent.from_dto(data, self.maze)
    
    async def run_loop(self):
        await asyncio.sleep(1)
        self.simulated_time.tick()
        print(f"round: {self.round_updates.current_round} time: {self.simulated_time.as_string()}")
        for name, agent in self.agents.items():
            print(f"updating {name}")
            agent.update(self.simulated_time.as_string(), self.maze)
        
        self.round_updates.add(self.simulated_time, self.agents)

        return self.round_updates.last

def main():
    round_updates = RoundUpdateSnapshots()
    simulation = Simulation(round_updates)

    api.start(simulation.run_loop, simulation.spawn_agent)

if __name__ == '__main__':
    main()