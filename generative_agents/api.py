import asyncio
from typing import List
import uuid
from aiohttp import web
from pydantic import BaseModel
import socketio


class MovementDTO(BaseModel):
    col: int
    row: int

class AgentDTO(BaseModel):
    name: str
    description: str
    location: str
    emoji: str
    activity: str
    movement: MovementDTO

class RoundUpdateDTO(BaseModel):
    round: int
    time: str
    agents: List[AgentDTO]

sids = set()

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

@sio.event
async def watch(sid):
    print("server received message!", sid)
    sids.add(sid)


round_updates = [
    RoundUpdateDTO(round=0, time="", agents=
                   [
                       AgentDTO(name="Klaus_Mueller", movement=MovementDTO(col=127, row=45), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Maria_Lopez", movement=MovementDTO(col=124, row=57), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Tom_Moreno", movement=MovementDTO(col=73, row=14), description="xyz", location="here", emoji="xyz", activity="walking")
                   ]),
    RoundUpdateDTO(round=1, time="", agents=
                   [
                       AgentDTO(name="Klaus_Mueller", movement=MovementDTO(col=127, row=44), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Maria_Lopez", movement=MovementDTO(col=125, row=57), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Tom_Moreno", movement=MovementDTO(col=74, row=14), description="xyz", location="here", emoji="xyz", activity="walking")
                   ]),
    RoundUpdateDTO(round=2, time="", agents=
                   [
                       AgentDTO(name="Klaus_Mueller", movement=MovementDTO(col=127, row=43), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Maria_Lopez", movement=MovementDTO(col=124, row=57), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Tom_Moreno", movement=MovementDTO(col=75, row=14), description="xyz", location="here", emoji="xyz", activity="walking"),
                   ]),
    RoundUpdateDTO(round=3, time="", agents=
                   [
                       AgentDTO(name="Klaus_Mueller", movement=MovementDTO(col=127, row=44), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Maria_Lopez", movement=MovementDTO(col=123, row=57), description="xyz", location="here", emoji="xyz", activity="walking"),
                       AgentDTO(name="Tom_Moreno", movement=MovementDTO(col=75, row=15), description="xyz", location="here", emoji="xyz", activity="walking")
                   ]),
    RoundUpdateDTO(round=4, time="", agents=
                [
                    AgentDTO(name="Klaus_Mueller", movement=MovementDTO(col=127, row=45), description="xyz", location="here", emoji="xyz", activity="walking"),
                    AgentDTO(name="Maria_Lopez", movement=MovementDTO(col=123, row=56), description="xyz", location="here", emoji="xyz", activity="walking"),
                    AgentDTO(name="Tom_Moreno", movement=MovementDTO(col=73, row=16), description="xyz", location="here", emoji="xyz", activity="walking")
                ])
]

# web socket implementaiton for spawning agents and sends back the assigned uuid4
@sio.event
async def spawn(sid, data):
    print("server received message!", sid)
    print(data)

    return str(uuid.uuid4())


async def updater():
    round_counter = 0 
    direction = 1

    while True:        
        await asyncio.sleep(5)
        emmited = False
        for sid in sids:
            await sio.emit('update', round_updates[round_counter].model_dump_json(), to=sid)
            emmited = True

        if emmited:
            if round_counter + direction >= len(round_updates):
                direction = -1
            
            if round_counter + direction < 0:
                direction = 1

            round_counter += direction



async def init_app():
    sio.start_background_task(updater)
    return app

if __name__ == '__main__':
    web.run_app(init_app())