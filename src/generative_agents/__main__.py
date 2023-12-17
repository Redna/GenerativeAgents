import asyncio
from typing import List
from generative_agents import global_state

from generative_agents.communication import api
from generative_agents.communication.models import AgentDTO, RoundUpdateDTO
from generative_agents.core.agent import Agent
from generative_agents.persistence.database import initialize_database
from generative_agents.simulation.maze import Maze
from generative_agents.simulation.time import SimulationTime


class RoundUpdateSnapshots():
    def __init__(self):
        self.rounds = []

    def add(self, time, agents: List[Agent]):
        agents_dto = [agent.to_dto() for agent in agents.values()]

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
        initialize_database(True)

        self.agents["Giorgio Rossi"] = Agent(name="Giorgio Rossi",
                                          time=global_state.time,
                                          description="Giorgio Rossi, known for his warm and attentive service, is a popular waiter at Hobbs Cafe, a local favorite for both its ambiance and cuisine. When he's not bustling around the cafe, Giorgio indulges in his love for reading, often losing himself in the pages of a good book. He also enjoys taking long, leisurely walks through the village, embracing the tranquility and charm of his surroundings, a perfect contrast to the lively atmosphere of the cafe. Giorgio's simple pleasures and dedication to his job make him a well-regarded member of the community.",
                                          innate_traits=["Easy going", "Competitive", "Confident"],
                                          age=25,
                                          location="the Ville:Giorgio Rossi's apartment:bathroom:shower",
                                          emoji="🤖",
                                          activity="idle",
                                          tile=self.maze.address_tiles["the Ville:Giorgio Rossi's apartment:bathroom:shower"][0])
        
        self.agents["John Lin"] = Agent(name="John Lin",
                                          time=global_state.time,
                                          description="John Lin, a dedicated pharmacist, is a familiar face at the Willows Market and Pharmacy, where his expertise and friendly demeanor are well appreciated by the community. Outside of work, he cherishes time with his family, including his wife, Mei Lin, and their son, Eddy. Although Mein and Eddy are currently enjoying a vacation in Alfter near Bonn in Germany. John is likely missing his favorite coffee from Hobbs Cafe, a testament to his love for their unique brews.",
                                          innate_traits=["Easy going", "Competitive", "Confident"],
                                          age=25,
                                          location="the Ville:Lin family's house:Mei and John Lin's bedroom",
                                          emoji="🤖",
                                          activity="idle",
                                          tile=self.maze.address_tiles["the Ville:Giorgio Rossi's apartment:bathroom:toilet"][0])
                                          #tile=self.maze.address_tiles["the Ville:Lin family's house:Mei and John Lin's bedroom"][0])
        
        
        self.round_updates = round_updates

    def spawn_agent(self, data: AgentDTO):
        print(
            f"spawning agent {data.name}, at {data.movement.col}, {data.movement.row}")
        self.agents[data.name] = Agent.from_dto(data, self.maze, self.simulated_time)

    async def run_loop(self):
        await asyncio.sleep(0.01)
        global_state.time.tick()
        print(
            f"round: {self.round_updates.current_round} time: {global_state.time.as_string()}")
        
        updates = []
        for name, agent in self.agents.items():
            print(f"scheduling update for {name}")
            updates.append(agent.update(global_state.time, self.maze, self.agents))

        updates = await asyncio.gather(*updates)

        for agent, tile in zip(self.agents.values(), updates):
            old_tile = agent.scratch.tile

            while not agent.scratch.finished_action_queue.empty():
                action = agent.scratch.finished_action_queue.get()
                if action.event.subject in old_tile.events:
                    del old_tile.events[action.event.subject]

            event = agent.scratch.action.event
            tile.events[event.subject] = agent.scratch.action.event
            
            object_action = agent.scratch.action.object_action
            if object_action and object_action.event:
                object_event = object_action.event
                if object_action.address in self.maze.address_tiles:
                    self.maze.address_tiles[object_action.address][0].events[object_event.subject] = object_event
                else:
                    print(f"WARNING: {object_action.address} not in maze")

            agent.scratch.tile = tile
            
            print(agent.name.center(80, "-"))
            if old_tile != tile:
                print(f"{agent.scratch.name} moved from {old_tile} to {tile}")
            else:
                print(f"{agent.scratch.name} is still at {tile}")
            print(f"{agent.scratch.name} is {agent.emoji}")
            print(f"{agent.scratch.name} is {agent.description}")

        self.round_updates.add(global_state.time, self.agents)

        return self.round_updates.last


def main():
    round_updates = RoundUpdateSnapshots()
    simulation = Simulation(round_updates)
    api.start(simulation.run_loop, simulation.spawn_agent)


if __name__ == '__main__':
    main()
