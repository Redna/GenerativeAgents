from typing import Callable, Coroutine
from aiohttp import web
import socketio


sids = set()
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

spawn_agent: Callable = None
update_simulation: Coroutine = None

@sio.event
async def spawn(sid, data: AgentDTO):
    if not spawn_agent:
        raise Exception("spawn_agent_function not set")
    
    spawn_agent(AgentDTO(**data))

@sio.event
def watch(sid):
    print("Client attached to server", sid)
    sids.add(sid)

async def updater():
    while True:   
        update = await update_simulation()
        # emit pydantic model as json dict
        await sio.emit('update', update.dict())     

async def init_app():
    sio.start_background_task(updater)
    return app

def start(update: Callable, spawn_agent_function: Callable):
    global spawn_agent
    global update_simulation
    spawn_agent = spawn_agent_function
    update_simulation = update
    web.run_app(init_app())
    