import asyncio
import sys

from loguru import logger

from generative_agents.common import global_state

# Configure Loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)

_sio_instance = None
_main_loop = None
_connected_sids = None


def initialize_socket_logging(sio, connected_sids: set):
    """
    Initialize the socket instance for logging broadcasting.
    Must be called from the main asyncio thread.
    """
    global _sio_instance, _main_loop, _connected_sids
    _sio_instance = sio
    _connected_sids = connected_sids
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = asyncio.get_event_loop()


def log_agent(agent_name: str, message: str, level: str = "INFO"):
    """
    Log an agent's thought or action, and optionally broadcast it over websocket.
    """
    if global_state.time:
        time_str = global_state.time.as_string()
    else:
        time_str = "UNKNOWN_TIME"

    formatted_message = (
        f"{time_str:<15} - {global_state.tick:<6}  {agent_name:<14}: {message}"
    )

    # Log to console
    if level == "INFO":
        logger.info(formatted_message)
    elif level == "DEBUG":
        logger.debug(formatted_message)
    elif level == "WARNING":
        logger.warning(formatted_message)
    elif level == "ERROR":
        logger.error(formatted_message)
    else:
        logger.info(formatted_message)

    # Broadcast via Websocket if initialized
    # Use run_coroutine_threadsafe because log_agent may be called from
    # worker threads (via asyncio.to_thread in the simulation loop).
    if _sio_instance and _main_loop and _connected_sids:
        future = asyncio.run_coroutine_threadsafe(
            _emit_log(agent_name, message, level, time_str), _main_loop
        )

        def _on_done(f):
            exc = f.exception()
            if exc:
                print(f"LOG_EMIT_ERROR: {exc}")

        future.add_done_callback(_on_done)


async def _emit_log(agent_name: str, message: str, level: str, time_str: str):
    """
    Emit the log to all connected watchers.
    The frontend filters by selected agent.
    """
    payload = {
        "agent": agent_name,
        "message": message,
        "level": level,
        "timestamp": time_str,
        "tick": global_state.tick,
    }

    sid_list = list(_connected_sids) if _connected_sids else []
    if not sid_list:
        return

    try:
        for sid in sid_list:
            await _sio_instance.emit("agent_log", payload, to=sid)
    except Exception as e:
        print(f"ERROR: Failed to emit log: {e}")
