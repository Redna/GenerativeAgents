import asyncio
import random
import socketio
from aiohttp import web
from datetime import datetime

# Import models from the project to ensure compatibility
# Make sure PYTHONPATH includes src/
from generative_agents.communication.models import AgentDTO, RoundUpdateDTO, MovementDTO

# Create Async SocketIO Server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
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
        "movement": {"col": 127, "row": 46}
    },
    {
        "name": "Maria_Lopez",
        "age": 22,
        "description": "A cheerful artist.",
        "location": "the Ville:artist's co-living space:Abigail Chen's room",
        "emoji": "🎨",
        "activity": "painting",
        "movement": {"col": 127, "row": 54}
    }
]

SIMULATION_ROUND = 0

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def watch(sid):
    print(f"Client {sid} started watching.")

@sio.event
async def spawn(sid, data):
    print(f"Client requested spawn: {data}")

async def mock_simulation_loop():
    global SIMULATION_ROUND
    print("Mock simulation loop started.")
    while True:
        SIMULATION_ROUND += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        agent_dtos = []
        for agent_data in AGENTS:
            # Random Movement
            move = random.choice([(0,1), (0,-1), (1,0), (-1,0), (0,0)])
            agent_data["movement"]["col"] += move[0]
            agent_data["movement"]["row"] += move[1]
            
            # Bounds check (simplified)
            agent_data["movement"]["col"] = max(0, min(agent_data["movement"]["col"], 200)) # adjust bounds as needed
            agent_data["movement"]["row"] = max(0, min(agent_data["movement"]["row"], 200))

            dto = AgentDTO(
                name=agent_data["name"],
                age=agent_data["age"],
                inniate_traits=[],
                description=agent_data["description"],
                location=agent_data["location"],
                emoji=agent_data["emoji"],
                activity=agent_data["activity"],
                movement=MovementDTO(col=agent_data["movement"]["col"], row=agent_data["movement"]["row"])
            )
            agent_dtos.append(dto)
        
        update = RoundUpdateDTO(
            round=SIMULATION_ROUND,
            time=current_time,
            agents=agent_dtos
        )
        
        # print(f"Emitting round {SIMULATION_ROUND} update...")
        await sio.emit('update', update.dict())
        await asyncio.sleep(1)

async def start_background_tasks(app):
    app['mock_loop'] = asyncio.create_task(mock_simulation_loop())

app.on_startup.append(start_background_tasks)

if __name__ == '__main__':
    print("Starting Mock Server on http://localhost:8000")
    web.run_app(app, port=8000)
