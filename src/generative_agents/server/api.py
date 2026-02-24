import asyncio
from typing import Callable

import socketio
from aiohttp import web

from generative_agents.common.models import AgentDTO

sids = set()

# Use AsyncServer with aiohttp
sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })
    try:
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    except web.HTTPException as ex:
        ex.headers["Access-Control-Allow-Origin"] = "*"
        ex.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        ex.headers["Access-Control-Allow-Headers"] = "Content-Type"
        raise

app = web.Application(middlewares=[cors_middleware])
sio.attach(app)


spawn_agent: Callable = None
update_simulation: Callable = None
get_agent_xray_fn: Callable = None
query_agent_memory_fn: Callable = None
pause_simulation_fn: Callable = None
resume_simulation_fn: Callable = None
get_simulation_status_fn: Callable = None


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
                await sio.emit("update", update.dict())

        await asyncio.sleep(0.01)


async def handle_xray(request):
    agent_name = request.query.get("agent")
    if not agent_name:
        return web.json_response({"error": "agent parameter is required"}, status=400)
        
    if get_agent_xray_fn:
        # Prevent blocking the main aiohttp thread by offloading synchronous AI retrieval
        xray = await asyncio.to_thread(get_agent_xray_fn, agent_name)
        if xray is None:
            return web.json_response({"error": "Agent not found"}, status=404)
        return web.json_response(xray)
    return web.json_response({"error": "API not ready"}, status=503)


async def handle_memory_query(request):
    agent_name = request.query.get("agent")
    query = request.query.get("q", "")
    limit = int(request.query.get("limit", 10))
    
    if not agent_name:
        return web.json_response({"error": "agent parameter is required"}, status=400)
        
    if query_agent_memory_fn:
        memories = await asyncio.to_thread(query_agent_memory_fn, agent_name, query, limit)
        return web.json_response(memories)
    return web.json_response({"error": "API not ready"}, status=503)


async def handle_pause(request):
    if pause_simulation_fn:
        pause_simulation_fn()
        return web.json_response({"status": "Simulation paused"})
    return web.json_response({"error": "API not ready"}, status=503)


async def handle_resume(request):
    if resume_simulation_fn:
        resume_simulation_fn()
        return web.json_response({"status": "Simulation resumed"})
    return web.json_response({"error": "API not ready"}, status=503)


async def handle_status(request):
    if get_simulation_status_fn:
        status = get_simulation_status_fn()
        return web.json_response({"paused": status})
    return web.json_response({"error": "API not ready"}, status=503)


async def handle_options(request):
    return web.Response()

def init_app(
    update: Callable, 
    spawn_agent_function: Callable,
    get_agent_xray: Callable = None,
    query_agent_memory: Callable = None,
    pause_sim: Callable = None,
    resume_sim: Callable = None,
    get_sim_status: Callable = None
):
    global spawn_agent
    global update_simulation
    global get_agent_xray_fn
    global query_agent_memory_fn
    global pause_simulation_fn
    global resume_simulation_fn
    global get_simulation_status_fn
    
    spawn_agent = spawn_agent_function
    update_simulation = update
    get_agent_xray_fn = get_agent_xray
    query_agent_memory_fn = query_agent_memory
    pause_simulation_fn = pause_sim
    resume_simulation_fn = resume_sim
    get_simulation_status_fn = get_sim_status

    # Initialize Logging with SIO instance
    from generative_agents.common.logging import initialize_socket_logging, log_agent

    initialize_socket_logging(sio, sids)
    log_agent("System", "Real Backend API Initialized", "INFO")

    # Start the background task
    app.on_startup.append(start_background_tasks)
    
    # Add HTTP routes
    app.router.add_options('/api/xray', handle_options)
    app.router.add_get('/api/xray', handle_xray)
    app.router.add_options('/api/memory', handle_options)
    app.router.add_get('/api/memory', handle_memory_query)
    app.router.add_options('/api/pause', handle_options)
    app.router.add_post('/api/pause', handle_pause)
    app.router.add_options('/api/resume', handle_options)
    app.router.add_post('/api/resume', handle_resume)
    app.router.add_options('/api/status', handle_options)
    app.router.add_get('/api/status', handle_status)
    
    return app


async def start_background_tasks(app):
    app["updater"] = asyncio.create_task(updater())


def get_app(
    update: Callable, 
    spawn_agent_function: Callable,
    get_agent_xray: Callable = None,
    query_agent_memory: Callable = None,
    pause_sim: Callable = None,
    resume_sim: Callable = None,
    get_sim_status: Callable = None
):
    return init_app(
        update, 
        spawn_agent_function, 
        get_agent_xray, 
        query_agent_memory,
        pause_sim,
        resume_sim,
        get_sim_status
    )
