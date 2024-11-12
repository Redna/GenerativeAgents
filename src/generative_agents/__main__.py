import asyncio
import os
import json
from time import sleep, time
from typing import Annotated, List, TypedDict

from langgraph.graph import StateGraph
from langgraph.constants import START, END, Send

from generative_agents import global_state

from generative_agents.communication import api
from generative_agents.communication.models import AgentDTO, RoundUpdateDTO
from generative_agents.core.agent import Agent
from generative_agents.core.agent_runner import AgentRunner, AgentRunnerState
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.persistence.database import initialize_database
from generative_agents.simulation.maze import Maze, BASE_PATH

ROUND_UPDATE = "round_update"
FAN_IN = "fan_in"
REFLECT_CHANGES = "reflect_changes"

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

def upsert(left: dict, right: dict):
    if left is None:
        left = {}
    if right is None:
        right = {}
    
    left.update(right)
    return left

class SimulationState(TypedDict):
    simulation_round: int
    agent_states: Annotated[dict[str, AgentRunnerState], upsert]

class Simulation():
    def __init__(self, round_updates: RoundUpdateSnapshots):
        self.maze = Maze()
        self.agents: dict[str, AgentRunner] = dict()
        self.__vision_start_tile = self.maze.get_random_tile()
        initialize_database(True)

        # load the agents file
        with open(os.path.join(BASE_PATH, "agents/agent_backstory.json"), "r") as f:
            agents = json.load(f)['agents']


        workflow = StateGraph(SimulationState)
        workflow.add_node(ROUND_UPDATE, self._round_update)
        workflow.add_node(REFLECT_CHANGES, self._reflect_changes)
        workflow.add_node(FAN_IN, self._fan_in)

        workflow.add_edge(START, ROUND_UPDATE)

        destinations = []
        self.initial_agent_states = {}
        for agent in agents:
            agent_runner = self.initialize_agent(name=agent['name'],
                                                               age=agent['age'],
                                                                innate_traits=agent['innate_traits'],
                                                                location=agent['location'],
                                                                emoji=agent['emoji'],
                                                                activity="idle",
                                                                description=agent['description'])

            self.agents[agent['name']] = agent_runner
            
            workflow.add_node(agent['name'], agent_runner.workflow.compile())
            workflow.add_edge(agent['name'], FAN_IN)
            destinations.append(agent['name'])
            self.initial_agent_states[agent['name']] = AgentRunnerState(
                agent_name=agent['name'],
                next_tile=agent['location'],
                perceived_events=[],
                retrieved=[],
                address=agent['location'],
                focused_event={},
                last_conversation=None
            )

        workflow.add_conditional_edges(ROUND_UPDATE, self._distribute_state, destinations) 
        workflow.add_edge(FAN_IN, REFLECT_CHANGES)
        workflow.add_conditional_edges(REFLECT_CHANGES, self._should_end, {True: END, False: ROUND_UPDATE})
        self.workflow = workflow

        self.round_updates = round_updates

    def stop(self):
        self.stopped = True

    async def run(self):
        self.stopped = False

        compiled_graph = self.workflow.compile()

        with open("graph.png", "wb") as f:
            f.write(compiled_graph.get_graph(xray=2).draw_mermaid_png())

        state = SimulationState(agent_states=self.initial_agent_states)

        compiled_graph.invoke(state)

    def initialize_agent(self, name, age, innate_traits, location, emoji, activity, description) -> AgentRunner: 
        agent = Agent(name=name, age=age, time=global_state.time, innate_traits=innate_traits, location=location, emoji=emoji, activity=activity, tile=self.maze.address_tiles[location][-1], tree=self.initialize_visible_memory_tree(), description=description)
        return AgentRunner(agent, self.maze, self.agents, global_state.time)

    def initialize_visible_memory_tree(self):
        tree = MemoryTree()
        for tile in self.maze.get_nearby_tiles(self.__vision_start_tile, 1000):
            tree.add(tile)
        return tree

    def spawn_agent(self, data: AgentDTO):
        print(
            f"spawning agent {data.name}, at {data.movement.col}, {data.movement.row}")
        self.agents[data.name] = Agent.from_dto(data, self.maze, self.simulated_time)

    
    def _round_update(self, state: SimulationState) -> SimulationState:
        print(f"round: {self.round_updates.current_round} time: {global_state.time.as_string()}")
        return SimulationState(simulation_round=global_state.tick)
    
    def _distribute_state(self, state: SimulationState) -> SimulationState:
        agent_states = state.get("agent_states", {})
        return [Send(name, agent_state) for name, agent_state in agent_states.items()]
    
    def _fan_in(self, state: AgentRunnerState) -> SimulationState:
        return SimulationState(agent_states={state.agent_name: state})

    def _reflect_changes(self, state: SimulationState) -> SimulationState:
        
        agent_states = state.get("agent_states", {})

        for name, agent_state in agent_states.items():
            next_tile = agent_state.get("next_tile", None)

            agent = self.agents[name].agent

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
            self.round_updates.add(global_state.time, self.agents)

            return state

    def _should_end(self, state: SimulationState) -> bool:
        return self.stop if self.stop else False

async def main():
    round_updates = RoundUpdateSnapshots()
    simulation = Simulation(round_updates)
    #api.start(simulation.run_loop, simulation.spawn_agent)
    
    await simulation.run()

if __name__ == '__main__':
    asyncio.run(main())
