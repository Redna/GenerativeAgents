import asyncio
import json
import os

from aiohttp import web

from generative_agents.agents.agent import Agent
from generative_agents.agents.memory.spatial import MemoryTree
from generative_agents.common import global_state
from generative_agents.persistence.database import initialize_database
from generative_agents.server import api
from generative_agents.simulation.engine import SimulationEngine
from generative_agents.simulation.maze import BASE_PATH, Maze


def initialize_visible_memory_tree(maze, start_tile):
    tree = MemoryTree()
    for tile in maze.get_nearby_tiles(start_tile, 1000):
        tree.add(tile)
    return tree


def initialize_agent(agent_data, maze, start_tile) -> Agent:
    name = agent_data["name"]
    location = agent_data["location"]

    # Ensure tile exists
    tile_coords = maze.address_tiles[location][-1]

    tree = initialize_visible_memory_tree(maze, start_tile)

    agent = Agent(
        name=name,
        age=agent_data["age"],
        time=global_state.time,
        innate_traits=agent_data["innate_traits"],
        location=location,
        emoji=agent_data["emoji"],
        activity="idle",
        tile=tile_coords,
        tree=tree,
        description=agent_data["description"],
    )
    return agent


async def main():
    from generative_agents.common.dspy_config import configure_dspy

    configure_dspy()

    maze = Maze()
    initialize_database(True)

    # Load agents
    with open(os.path.join(BASE_PATH, "agents/agent_backstory.json"), "r") as f:
        agent_data_list = json.load(f)["agents"]

    # Initial vision logic (from original code)
    vision_start_tile = maze.get_random_tile()

    agents = []
    for agent_data in agent_data_list:
        agent = initialize_agent(agent_data, maze, vision_start_tile)
        agents.append(agent)

    # Initialize Engine
    engine = SimulationEngine(maze, agents, global_state.time)

    # Initialize the API app
    app = api.get_app(engine.get_latest_round_update, engine.spawn_agent)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8000)
    await site.start()

    print("Server started at http://localhost:8000")

    # Run Simulation Loop
    try:
        while True:
            # Execute step
            engine.step()

            # Print status (optional, matching previous output style)
            # print(f"Time: {global_state.time.as_string()}")

            # Yield control to asyncio event loop to allow API requests processing
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Simulation stopped with error: {e}")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
