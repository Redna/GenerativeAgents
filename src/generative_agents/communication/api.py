from dataclasses import asdict
from typing import Callable, Coroutine, Dict, Set, Tuple
from aiohttp import web
import socketio
import asyncio

from .models import AgentDTO

sids = set()

# Use AsyncServer with aiohttp
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)


spawn_agent: Callable = None
update_simulation: Callable = None

@sio.event
async def spawn(sid, data: AgentDTO):
    if not spawn_agent:
        raise Exception("spawn_agent_function not set")
    
    # Assuming spawn_agent might need to be awaited or just called
    # If spawn_agent is async, await it. If not, just call it.
    if asyncio.iscoroutinefunction(spawn_agent):
        await spawn_agent(AgentDTO(**data))
    else:
        spawn_agent(AgentDTO(**data))

@sio.event
async def watch(sid):
    print("Client attached to server", sid)
    sids.add(sid)

@sio.event
async def connect(sid, environ):
    print("Client connected", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected", sid)
    if sid in sids:
        sids.remove(sid)

async def updater():
    while True:
        if update_simulation:
            # If update_simulation is async, await it
            if asyncio.iscoroutinefunction(update_simulation):
                update = await update_simulation()
            else:
                update = update_simulation()
            
            # emit pydantic model as json dict
            if update:
                 await sio.emit('update', update.dict())
        
        await asyncio.sleep(0.01)

def init_app(update: Callable, spawn_agent_function: Callable):
    global spawn_agent
    global update_simulation
    spawn_agent = spawn_agent_function
    update_simulation = update
    
    # Start the background task
    # We can't use sio.start_background_task here easily because we need the loop
    # Ideally, the caller should start the background task or we use on_startup
    app.on_startup.append(start_background_tasks)
    return app

async def start_background_tasks(app):
    app['updater'] = asyncio.create_task(updater())

def get_app(update: Callable, spawn_agent_function: Callable):
    return init_app(update, spawn_agent_function)