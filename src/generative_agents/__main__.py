from time import sleep, time
from typing import List
from generative_agents import global_state

from generative_agents.communication import api
from generative_agents.communication.models import AgentDTO, RoundUpdateDTO
from generative_agents.core.agent import Agent, AgentRunner
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.persistence.database import initialize_database
from generative_agents.simulation.maze import Maze
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

        self.agents["Giorgio Rossi"] = self.initialize_agent(name="Giorgio Rossi",
                                            age=25,
                                            time=global_state.time,
                                            innate_traits=["Easy going", "Competitive", "Confident"],
                                            location="the Ville:Giorgio Rossi's apartment:main room:bed",
                                            emoji="🤖",
                                            activity="idle",
                                            tree=self.initialize_visible_memory_tree(),
                                            tile=self.maze.address_tiles["the Ville:Giorgio Rossi's apartment:main room:bed"][0],
                                            description="Giorgio Rossi, known for his warm and attentive service, is a popular waiter at Hobbs Cafe, a local favorite for both its ambiance and cuisine. When he's not bustling around the cafe, Giorgio indulges in his love for reading, often losing himself in the pages of a good book. He also enjoys taking long, leisurely walks through the village, embracing the tranquility and charm of his surroundings, a perfect contrast to the lively atmosphere of the cafe. Giorgio's simple pleasures and dedication to his job make him a well-regarded member of the community."
                                            )

        self.agents["John Lin"] = self.initialize_agent(name="John Lin",
                                            age=25,
                                            time=global_state.time,
                                            innate_traits=["Easy going", "Competitive", "Confident"],
                                            location="the Ville:Lin family's house:Mei and John Lin's bedroom",
                                            emoji="🤖",
                                            activity="idle",
                                            tree=self.initialize_visible_memory_tree(),
                                            tile=self.maze.address_tiles["the Ville:Lin family's house:Mei and John Lin's bedroom"][0],
                                            description="John Lin, a dedicated pharmacist, is a familiar face at the Willows Market and Pharmacy, where his expertise and friendly demeanor are well appreciated by the community. Outside of work, he cherishes time with his family, including his wife, Mei Lin, and their son, Eddy. Although Mein and Eddy are currently enjoying a vacation in Alfter near Bonn in Germany. John is likely missing his favorite coffee from Hobbs Cafe, a testament to his love for their unique brews.")

        self.agents["Klaus Mueller"] = self.initialize_agent(name="Klaus Mueller",
                                            age=21,
                                            time=global_state.time,
                                            innate_traits=["openness", "extraversion"],
                                            location="the Ville:Dorm for Oak Hill College:Klaus Mueller's room",
                                            emoji="🤖",
                                            activity="idle",
                                            tree=self.initialize_visible_memory_tree(),
                                            tile=self.maze.address_tiles["the Ville:Dorm for Oak Hill College:Klaus Mueller's room"][0],
                                            description="Klaus Mueller, a talented writer, is a student at Oak Hill College, where he studies literature and creative writing. His room is filled with stacks of books and journals, as he spends hours crafting stories and poems that reflect his unique perspective on the world. Klaus's passion for writing is matched only by his love for his friends and family, who often gather in his room to share their own stories and ideas.")


        self.agents["Maria Lopez"] = self.initialize_agent(name="Maria Lopez",
                                            age=22,
                                            time=global_state.time,
                                            innate_traits=["Easy going", "Competitive", "Confident"],
                                            location="the Ville:Dorm for Oak Hill College:Maria Lopez's room",
                                            emoji="🤖",
                                            activity="idle",
                                            tree=self.initialize_visible_memory_tree(),
                                            tile=self.maze.address_tiles["the Ville:Dorm for Oak Hill College:Maria Lopez's room"][0],
                                            description="Maria Lopez, a talented musician, is a student at Oak Hill College, where she studies music theory and composition. Her room is filled with the sounds of her guitar and piano, as she practices new songs and melodies late into the night. Maria's passion for music is matched only by her love for her friends and family, who often gather in her room to listen to her latest compositions and share stories of their own.")

        self.agents["Isabella Rodriguez"] = self.initialize_agent(name="Isabella Rodriguez",
                                            age=28,
                                            time=global_state.time,
                                            innate_traits=["Easy going", "Competitive", "Confident"],
                                            location="the Ville:Isabella Rodriguez's apartment:main room",
                                            emoji="🤖",
                                            activity="idle",
                                            tree=self.initialize_visible_memory_tree(),
                                            #the Ville, Isabella Rodriguez's apartment, main room, sp-A
                                            tile=self.maze.address_tiles["the Ville:Isabella Rodriguez's apartment:main room"][0],
                                            description="Isabella Rodriguez, a talented artist, is known for her vibrant and expressive paintings that capture the beauty of the Ville and its residents. Her studio is a cozy space filled with colorful canvases and the scent of fresh paint, where she spends hours creating new works of art. Isabella's passion for art is matched only by her love for her family and friends, who often inspire her creations with their unique personalities and stories.")
 
        self.round_updates = round_updates

    def initialize_agent(self, name, age, time, innate_traits, location, emoji, activity, tree, tile, description): 
        agent = Agent(name=name, age=age, time=time, innate_traits=innate_traits, location=location, emoji=emoji, activity=activity, tile=tile, tree=tree, description=description)
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
