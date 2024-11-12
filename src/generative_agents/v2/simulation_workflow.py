
import json
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from generative_agents.simulation.maze import BASE_PATH, Maze, Tile
from generative_agents.simulation.time import SimulationTime
from generative_agents.v2.agent_workflow import AgentState, AgentWorkflow
from generative_agents.v2.memory.spatial import MemoryTree


class messages(TypedDict):
    from_agent: str
    to_agent: str
    message: str


def update_agents(old: dict[str, AgentState], new: dict[str, AgentState]):
    if not old:
        return new

    if not new:
        return old

    agents = dict()

    for key, value in old.items():
        agents[key] = value

    for key, value in new.items():
        agents[key] = value

    return agents


class SimulationState(TypedDict):
    time: SimulationTime
    maze: Maze
    agents: Annotated[dict[str, AgentState], update_agents]


def initialize_visible_memory_tree(maze: Maze, vision_start_tile: Tile) -> MemoryTree:
    tree = MemoryTree()
    for tile in maze.get_nearby_tiles(vision_start_tile, 1000):
        tree.add(tile)
    return tree


def should_end(state: SimulationState) -> bool:
    return False


def set_time(state: SimulationState):
    pass


def update_maze(state: SimulationState):
    pass


def init_graph(agent_names: list[str]):
    simulation = StateGraph(SimulationState)

    simulation.add_node("set_time", set_time)
    simulation.add_node("update_maze", update_maze)

    simulation.add_edge(START, "set_time")

    for agent_name in agent_names:
        simulation.add_node(agent_name, AgentWorkflow())
        simulation.add_edge("set_time", agent_name)
        simulation.add_edge(agent_name, "update_maze")

    simulation.add_conditional_edges("update_maze", should_end, END)

    return simulation


def initialize_state() -> SimulationState:
    maze = Maze()
    simulation_time = SimulationTime(increment=1)

    with open(os.path.join(BASE_PATH, "agents/agent_backstory.json"), "r") as f:
        agent_backstories = json.load(f)['agents']

    agents = dict()

    for agent_backstory in agent_backstories:
        visible_memory_tree = initialize_visible_memory_tree(maze=maze,
                                                             vision_start_tile=maze.address_tiles[agent_backstory['location']][-1])
        agents[agent_backstory['name']] = AgentState(name=agent_backstory['name'],
                                                     age=agent_backstory['age'],
                                                     description=agent_backstory['description'],
                                                     innate_traits=agent_backstory['innate_traits'],
                                                     time=simulation_time,
                                                     location=agent_backstory['location'],
                                                     emoji=agent_backstory['emoji'],
                                                     activity="idle",
                                                     tile=maze.address_tiles[agent_backstory['location']][-1],
                                                     tree=visible_memory_tree)

    return SimulationState(time=simulation_time, maze=maze, agents=agents)


if __name__ == "__main__":
    state = initialize_state()

    simulation = init_graph(list(state['agents'].keys()))

    graph = simulation.compile(state)

    graph.get_graph().draw_png("simulation.png")
