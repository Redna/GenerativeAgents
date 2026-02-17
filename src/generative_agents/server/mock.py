import asyncio
import random
from datetime import datetime

import socketio
from aiohttp import web

# Import models from the project to ensure compatibility
# Make sure PYTHONPATH includes src/
from generative_agents.common.models import AgentDTO, MovementDTO, RoundUpdateDTO
from generative_agents.simulation.maze import Maze

# Create Async SocketIO Server
sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# Mock Data
AGENTS = [
    {
        "name": "Klaus_Mueller",
        "age": 20,
        "description": "A diligent student.",
        "location": "the Ville:Moreno family's house:common room",
        "emoji": "📚",
        "activity": "reading",
        "movement": {"col": 127, "row": 46},
        "path": [],
    },
    {
        "name": "Maria_Lopez",
        "age": 22,
        "description": "A cheerful artist.",
        "location": "the Ville:artist's co-living space:Abigail Chen's room",
        "emoji": "🎨",
        "activity": "painting",
        "movement": {"col": 127, "row": 54},
        "path": [],
    },
]

SIMULATION_ROUND = 0


@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


@sio.event
async def spawn(sid, data):
    print(f"Client {sid} spawned something... ignored.")


@sio.event
async def watch(sid):
    print(f"Client {sid} started watching.")


@sio.event
async def subscribe_agent_log(sid, data):
    agent_name = data.get("agent_name")
    if agent_name:
        print(f"Client {sid} subscribing to logs for {agent_name}")
        await sio.enter_room(sid, agent_name)


@sio.event
async def unsubscribe_agent_log(sid, data):
    agent_name = data.get("agent_name")
    if agent_name:
        print(f"Client {sid} unsubscribing from logs for {agent_name}")
        await sio.leave_room(sid, agent_name)


async def mock_simulation_loop():
    global SIMULATION_ROUND
    global SIMULATION_ROUND
    print("DEBUG: Mock simulation loop MOCKED_LOOP_START")
    print("Mock simulation loop started.")

    # Initialize Logging with SIO instance
    from generative_agents.common.logging import initialize_socket_logging, log_agent

    initialize_socket_logging(sio)
    log_agent("System", "Mock Simulation Initialized", "INFO")

    # Initialize Pathfinding
    try:
        maze = Maze()
        print(f"Maze '{maze.maze_name}' loaded for mock server pathfinding.")
    except Exception as e:
        print(f"Failed to load Maze: {e}")
        return

    # Define targets for agents
    targets = {
        "Klaus_Mueller": {
            "target_loc": (120, 20),  # Library (Oak Hill College:library)
            "target_activity": "reading at the library",
            "target_emoji": "📖",
            "reached": False,
        },
        "Maria_Lopez": {
            "target_loc": (24, 41),  # Park (Johnson Park:park)
            "target_activity": "painting in the park",
            "target_emoji": "🖌️",
            "reached": False,
        },
    }

    from generative_agents.common import global_state

    while True:
        SIMULATION_ROUND += 1
        global_state.tick = SIMULATION_ROUND
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        agent_dtos = []
        for agent_data in AGENTS:
            name = agent_data["name"]

            if name in targets:
                target = targets[name]
                curr_col = agent_data["movement"]["col"]
                curr_row = agent_data["movement"]["row"]
                dest_col, dest_row = target["target_loc"]

                # Pathfinding Logic
                # If we don't have a path, or we are not at target, calculate it
                if not agent_data.get("path") and not target["reached"]:
                    # Calculate path
                    try:
                        start_tile = maze.get_tile(curr_col, curr_row)
                        end_tile = maze.get_tile(dest_col, dest_row)
                        path = maze.find_path(start_tile, end_tile)
                        if path and len(path) > 1:
                            agent_data["path"] = path[1:]  # Exclude current tile
                            log_agent(
                                name,
                                f"Calculated path: {len(path)} steps to {target['target_activity']}",
                                "INFO",
                            )
                        else:
                            log_agent(
                                name,
                                f"No path found from ({curr_col},{curr_row}) to ({dest_col},{dest_row})",
                                "WARNING",
                            )
                    except Exception as e:
                        log_agent(name, f"Pathfinding error: {e}", "ERROR")

                # Move along path
                if agent_data.get("path"):
                    next_tile = agent_data["path"].pop(0)
                    agent_data["movement"]["col"] = next_tile.x
                    agent_data["movement"]["row"] = next_tile.y

                    # Log walking frequently
                    if SIMULATION_ROUND % 2 == 0:
                        log_agent(
                            name, f"Walking to {target['target_activity']}...", "INFO"
                        )

                # Check if reached target
                # We check distance to target or if path is empty and we are at target
                if (
                    agent_data["movement"]["col"] == dest_col
                    and agent_data["movement"]["row"] == dest_row
                ):
                    if not target["reached"]:
                        target["reached"] = True
                        agent_data["activity"] = target["target_activity"]
                        agent_data["emoji"] = target["target_emoji"]
                        description = (
                            f"Arrived at destination to {target['target_activity']}."
                        )
                        agent_data["description"] = description
                        log_agent(name, description, "INFO")
                        agent_data["path"] = []  # Clear path just in case
                else:
                    agent_data["activity"] = "walking"
            else:
                # Random movement for others (or implement random wander using maze later)
                # For now, keep simple random walk but check collision

                # Log wandering occasionally
                if SIMULATION_ROUND % 2 == 0:
                    log_agent(
                        name,
                        f"Wandering around {agent_data['location'].split(':')[-1]}...",
                        "INFO",
                    )

                move = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
                next_x = agent_data["movement"]["col"] + move[0]
                next_y = agent_data["movement"]["row"] + move[1]

                # Check collision
                if 0 <= next_x < maze.maze_width and 0 <= next_y < maze.maze_height:
                    tile = maze.get_tile(next_x, next_y)
                    if tile.is_walkable():
                        agent_data["movement"]["col"] = next_x
                        agent_data["movement"]["row"] = next_y

            # Simulate random thoughts (Global check, so it happens during walking OR wandering)
            if random.random() < 0.2:
                thoughts = [
                    "Typical day...",
                    "I wonder who I'll meet today.",
                    "The weather is nice.",
                    "Did I lock the door?",
                    "I should call my mom.",
                ]
                log_agent(name, f"Thought: {random.choice(thoughts)}", "DEBUG")

            # Bounds check (simplified)
            agent_data["movement"]["col"] = max(
                0, min(agent_data["movement"]["col"], 140)
            )
            agent_data["movement"]["row"] = max(
                0, min(agent_data["movement"]["row"], 100)
            )

            dto = AgentDTO(
                name=agent_data["name"],
                age=agent_data["age"],
                inniate_traits=[],
                description=agent_data["description"],
                location=agent_data["location"],
                emoji=agent_data["emoji"],
                activity=agent_data["activity"],
                movement=MovementDTO(
                    col=agent_data["movement"]["col"], row=agent_data["movement"]["row"]
                ),
            )
            agent_dtos.append(dto)

        update = RoundUpdateDTO(
            round=SIMULATION_ROUND, time=current_time, agents=agent_dtos
        )

        # print(f"Emitting round {SIMULATION_ROUND} update...")
        await sio.emit("update", update.dict())
        await asyncio.sleep(0.5)  # Speed up a bit for demo


async def start_background_tasks(app):
    app["mock_loop"] = asyncio.create_task(mock_simulation_loop())


app.on_startup.append(start_background_tasks)

if __name__ == "__main__":
    print("Starting Mock Server on http://localhost:8000")
    web.run_app(app, port=8000)
