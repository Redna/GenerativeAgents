# Class used for understanding the inner workings of the agent.
import asyncio
from enum import Enum
from typing import Callable, Dict, List

from generative_agents import global_state
from generative_agents.core.whisper.thought import Thought

emitter = None
min_level = 0

def whisper(agent: str, content: str, level: int = 0):
    if emitter:
        asyncio.get_running_loop().create_task(emitter(Thought(agent, content, level)))

    print(f"{global_state.time.as_string():<15} - {global_state.tick:<6}  {agent:<14}: {content}")