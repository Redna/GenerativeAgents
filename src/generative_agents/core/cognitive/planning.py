# controllers.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import Agent
    
class Planning:
    def __init__(self, agent: Agent):
        self.agent = agent
    
    
    
    