import asyncio
from typing import List

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
        initialize_database(True)
        self.maze = Maze()
        self.agents: List[Agent] = dict()
        self.simulated_time = SimulationTime(5)
        self.agents["Test Agent"] = Agent(name="Test Agent",
                                          time=self.simulated_time,
                                          description="A test agent",
                                          innate_traits=["Easy going", "Competitive", "Confident"],
                                          age=25,
                                          location="the Ville:Giorgio Rossi's apartment:bathroom:shower",
                                          emoji="🤖",
                                          activity="idle",
                                          tile=self.maze.address_tiles["the Ville:Giorgio Rossi's apartment:bathroom:shower"][0])
        self.round_updates = round_updates

    def spawn_agent(self, data: AgentDTO):
        print(
            f"spawning agent {data.name}, at {data.movement.col}, {data.movement.row}")
        self.agents[data.name] = Agent.from_dto(data, self.maze, self.simulated_time)

    async def run_loop(self):
        await asyncio.sleep(1)
        self.simulated_time.tick()
        print(
            f"round: {self.round_updates.current_round} time: {self.simulated_time.as_string()}")
        for name, agent in self.agents.items():
            print(f"updating {name}")
            await agent.update(self.simulated_time, self.maze, self.agents)

        self.round_updates.add(self.simulated_time, self.agents)

        return self.round_updates.last


def main():
    round_updates = RoundUpdateSnapshots()
    simulation = Simulation(round_updates)

    api.start(simulation.run_loop, simulation.spawn_agent)


if __name__ == '__main__':
    main()
