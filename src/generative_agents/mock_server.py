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
    
    # Define targets for agents
    targets = {
        "Klaus_Mueller": {
            "target_loc": (100, 80), # Library (mock coords)
            "target_activity": "reading at the library",
            "target_emoji": "📖",
            "reached": False
        },
        "Maria_Lopez": {
            "target_loc": (40, 60), # Park (mock coords)
            "target_activity": "painting in the park",
            "target_emoji": "🖌️",
            "reached": False
        }
    }

    while True:
        SIMULATION_ROUND += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        agent_dtos = []
        for agent_data in AGENTS:
            name = agent_data["name"]
            
            if name in targets:
                target = targets[name]
                curr_col = agent_data["movement"]["col"]
                curr_row = agent_data["movement"]["row"]
                dest_col, dest_row = target["target_loc"]

                # Simple movement logic
                if curr_col < dest_col:
                    agent_data["movement"]["col"] += 1
                elif curr_col > dest_col:
                    agent_data["movement"]["col"] -= 1
                
                if curr_row < dest_row:
                    agent_data["movement"]["row"] += 1
                elif curr_row > dest_row:
                    agent_data["movement"]["row"] -= 1

                # Check if reached
                if agent_data["movement"]["col"] == dest_col and agent_data["movement"]["row"] == dest_row:
                    if not target["reached"]:
                        target["reached"] = True
                        agent_data["activity"] = target["target_activity"]
                        agent_data["emoji"] = target["target_emoji"]
                        agent_data["description"] = f"Arrived at destination to {target['target_activity']}."
                else:
                     agent_data["activity"] = "walking"
            else:
                 # Random movement for others
                move = random.choice([(0,1), (0,-1), (1,0), (-1,0), (0,0)])
                agent_data["movement"]["col"] += move[0]
                agent_data["movement"]["row"] += move[1]

            
            # Bounds check (simplified)
            agent_data["movement"]["col"] = max(0, min(agent_data["movement"]["col"], 140)) 
            agent_data["movement"]["row"] = max(0, min(agent_data["movement"]["row"], 100))

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
        await asyncio.sleep(0.5) # Speed up a bit for demo

async def start_background_tasks(app):
    app['mock_loop'] = asyncio.create_task(mock_simulation_loop())

app.on_startup.append(start_background_tasks)

if __name__ == '__main__':
    print("Starting Mock Server on http://localhost:8000")
    web.run_app(app, port=8000)
