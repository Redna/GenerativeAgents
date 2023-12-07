from dataclasses import asdict
from typing import Callable, Coroutine, Dict, Set, Tuple
from aiohttp import web
import socketio

from generative_agents.core.whisper.thought import Thought
from generative_agents.core.whisper import whisper

from .models import AgentDTO


sids = set()

whisper_listeners: Set[Tuple] = set()


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

@sio.event
async def listen_to_whisper(sid, data):
    global whisper_listeners
    print(f"Client {sid} listening to {data['agent']} at level {data['level']}")
    whisper_listeners.add((sid, data['agent'], data["level"]))

async def whisper_emitter(thought: Thought):
    global whisper_listeners
    for sid, agent, level in whisper_listeners:
        if agent == thought.agent and level >= thought.level:
            await sio.emit('whisper', data=thought.__dict__, room=sid)

async def updater():
    while True:   
        update = await update_simulation()
        # emit pydantic model as json dict
        await sio.emit('update', update.dict())     

async def init_app():
    whisper.emitter = whisper_emitter
    sio.start_background_task(updater)
    return app

def start(update: Callable, spawn_agent_function: Callable):
    global spawn_agent
    global update_simulation
    spawn_agent = spawn_agent_function
    update_simulation = update
    web.run_app(init_app())
    