from dataclasses import asdict
from typing import Callable, Coroutine, Dict, Set, Tuple
from aiohttp import web
import socketio

from .models import AgentDTO

from gevent import pywsgi

sids = set()

sio = socketio.Server(async_mode='gevent', cors_allowed_origins='*')
app = socketio.WSGIApp(sio)


spawn_agent: Callable = None
update_simulation: Callable = None

@sio.event
def spawn(sid, data: AgentDTO):
    if not spawn_agent:
        raise Exception("spawn_agent_function not set")
    
    spawn_agent(AgentDTO(**data))

@sio.event
def watch(sid):
    print("Client attached to server", sid)
    sids.add(sid)

def updater():
    while True:   
        update = update_simulation()
        # emit pydantic model as json dict
        sio.emit('update', update.dict())
        sio.sleep(0.01)

def init_app():
    sio.start_background_task(updater)
    return app

def start(update: Callable, spawn_agent_function: Callable):
    global spawn_agent
    global update_simulation
    spawn_agent = spawn_agent_function
    update_simulation = update
    pywsgi.WSGIServer(('', 8000), init_app()).serve_forever()