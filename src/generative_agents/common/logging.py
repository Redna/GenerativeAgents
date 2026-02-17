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


def initialize_socket_logging(sio):
    """
    Initialize the socket instance for logging broadcasting.
    """
    global _sio_instance
    _sio_instance = sio


def log_agent(agent_name: str, message: str, level: str = "INFO"):
    """
    Log an agent's thought or action, and optionally broadcast it over websocket.
    """
    # Format similar to old whisper: {time} - {tick} {agent}: {message}
    # We use loguru's formatting for the timestamp, so we just include the rest
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
    if _sio_instance:
        asyncio.create_task(_emit_log(agent_name, message, level, time_str))


async def _emit_log(agent_name: str, message: str, level: str, time_str: str):
    """
    Emit the log to the specific agent room.
    """
    payload = {
        "agent": agent_name,
        "message": message,
        "level": level,
        "timestamp": time_str,
        "tick": global_state.tick,
    }

    try:
        # Broadcast log to all clients (namespace '/')
        # Frontend ensures only the selected agent's logs are shown
        await _sio_instance.emit("agent_log", payload, room=agent_name, namespace="/")
    except Exception as e:
        print(f"ERROR: Failed to emit log to {agent_name}: {e}")
