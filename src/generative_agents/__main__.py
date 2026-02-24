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


def initialize_agent(agent_data, maze, start_tile) -> Agent:
    name = agent_data["name"]
    location = agent_data["location"]

    # Ensure tile exists
    tile_coords = maze.address_tiles[location][-1]

    agent = Agent(
        name=name,
        age=agent_data["age"],
        time=global_state.time,
        innate_traits=agent_data["innate_traits"],
        location=location,
        emoji=agent_data["emoji"],
        activity="idle",
        tile=tile_coords,
        description=agent_data["description"],
    )
    return agent


async def main():
    from generative_agents.common.dspy_config import configure_dspy

    configure_dspy()

    maze = Maze()
    initialize_database(True)

    from generative_agents.common.utils import get_project_root
    import glob
    import pickle
    
    checkpoints_dir = os.path.join(get_project_root(), "storage", "checkpoints")
    latest_checkpoint = None
    
    if os.path.exists(checkpoints_dir):
        files = glob.glob(os.path.join(checkpoints_dir, "*.pkl"))
        if files:
            latest_checkpoint = max(files, key=os.path.getctime)

    if latest_checkpoint:
        print(f"Loading simulation state from {latest_checkpoint}...")
        with open(latest_checkpoint, "rb") as f:
            state = pickle.load(f)
            
        agents = list(state["agents"].values())
        engine = SimulationEngine(maze, agents, state["time"])
        engine.round_updates = state.get("round_updates", [])
        
        print(f"Resumed {len(agents)} agents at {state['time'].as_string()}")
    else:
        print("Starting fresh simulation...")
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
    app = api.get_app(
        engine.get_latest_round_update, 
        engine.spawn_agent,
        engine.get_agent_xray,
        engine.query_agent_memory,
        engine.pause,
        engine.resume,
        lambda: engine.is_paused
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8000)
    await site.start()

    print("Server started at http://localhost:8000")

    # Run Simulation Loop
    try:
        while True:
            # Execute step in a separate thread so sync DSPy calls don't block socket.io
            if not engine.is_paused:
                await asyncio.to_thread(engine.step)

            # Yield control to asyncio event loop to allow API requests processing
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Simulation stopped with error: {e}")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
