import os
import json
from time import sleep, time
from typing import List
from generative_agents import global_state

from generative_agents.communication import api
from generative_agents.communication.models import AgentDTO, RoundUpdateDTO
from generative_agents.core.agent import Agent, AgentRunner
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.persistence.database import initialize_database
from generative_agents.simulation.maze import Maze, BASE_PATH
from generative_agents.simulation.time import SimulationTime


class RoundUpdateSnapshots():
    def __init__(self):
        self.rounds = []

    def add(self, time, agents: List[Agent]):
        agents_dto = [agent_runner.agent.to_dto() for agent_runner in agents.values()]

        converted_date_time = time.as_string()
        round_update = RoundUpdateDTO(
            round=len(self.rounds), time=converted_date_time, agents=agents_dto)
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



class Simulation():
    def __init__(self, round_updates: RoundUpdateSnapshots):
        self.maze = Maze()
        self.agents: List[Agent] = dict()
        self.__vision_start_tile = self.maze.get_random_tile()
        initialize_database(True)

        # load the agents file
        with open(os.path.join(BASE_PATH, "agents/agent_backstory.json"), "r") as f:
            agents = json.load(f)['agents']

        for agent in agents:
            self.agents[agent['name']] = self.initialize_agent(name=agent['name'],
                                                               age=agent['age'],
                                                                innate_traits=agent['innate_traits'],
                                                                location=agent['location'],
                                                                emoji=agent['emoji'],
                                                                activity="idle",
                                                                description=agent['description'])
        #[tile for tile in maze.address_tiles if "the Ville:artist's co-living space:Abigail Chen" in tile]
 
        self.round_updates = round_updates

    def initialize_agent(self, name, age, innate_traits, location, emoji, activity, description): 
        agent = Agent(name=name, age=age, time=global_state.time, innate_traits=innate_traits, location=location, emoji=emoji, activity=activity, tile=self.maze.address_tiles[location][-1], tree=self.initialize_visible_memory_tree(), description=description)
        return AgentRunner(agent)

    def initialize_visible_memory_tree(self):
        tree = MemoryTree()
        for tile in self.maze.get_nearby_tiles(self.__vision_start_tile, 1000):
            tree.add(tile)
        return tree


    def spawn_agent(self, data: AgentDTO):
        print(
            f"spawning agent {data.name}, at {data.movement.col}, {data.movement.row}")
        self.agents[data.name] = Agent.from_dto(data, self.maze, self.simulated_time)

    def run_loop(self):
        global_state.time.tick()
        print(
            f"round: {self.round_updates.current_round} time: {global_state.time.as_string()}")
        
        agents = [agent.agent for agent in self.agents.values()]

        for name, agent_runner in self.agents.items():
            start = time()
            print(f"scheduling update for {name}")
            agent = agent_runner.agent
            next_tile = agent_runner.update(global_state.time, self.maze, agents)

            old_tile = agent.scratch.tile

            while agent.scratch.finished_action:
                action = agent.scratch.finished_action.pop(0)
                if action.event.subject in old_tile.events:
                    del old_tile.events[action.event.subject]

            event = agent.scratch.action.event
            next_tile.events[event.subject] = agent.scratch.action.event
            
            object_action = agent.scratch.action.object_action
            if object_action and object_action.event:
                object_event = object_action.event
                if object_action.address in self.maze.address_tiles:
                    self.maze.address_tiles[object_action.address][0].events[object_event.subject] = object_event
                else:
                    print(f"WARNING: {object_action.address} not in maze")

            agent.scratch.tile = next_tile
            
            print(agent.name.center(80, "-"))
            if old_tile != next_tile:
                print(f"{agent.scratch.name} moved from {old_tile} to {next_tile}")
            else:
                print(f"{agent.scratch.name} is still at {next_tile}")
            print(f"{agent.scratch.name} is {agent.emoji}")
            print(f"{agent.scratch.name} is {agent.description}")

            print("updated agent in: ", time() - start, " seconds")

        updated_agents = {name: agent_runner.agent for name, agent_runner in self.agents.items()}
        self.round_updates.add(global_state.time, self.agents)
        return self.round_updates.last


def main():
    round_updates = RoundUpdateSnapshots()
    simulation = Simulation(round_updates)
    api.start(simulation.run_loop, simulation.spawn_agent)
    #while(True):
    #    simulation.run_loop()

if __name__ == '__main__':
    main()
