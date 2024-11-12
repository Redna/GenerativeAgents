

import datetime
from typing import TypedDict
from generative_agents.simulation.maze import Tile
from generative_agents.simulation.time import SimulationTime
from generative_agents.v2.memory.spatial import MemoryTree
from generative_agents.v2.memory.scratch import Scratch

from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    name: str


class AgentWorkflow(StateGraph):
    def __init__(self) -> None:
        super().__init__(AgentState)
        self.add_node("perceive", self.perceive)
        self.add_node("retrieve", self.retrieve)
        self.add_node("plan", self.plan)
        self.add_node("execute", self.execute)
        self.add_node("reflect", self.reflect)
        
        self.add_edge(START, "perceive")
        self.add_edge("perceive", "retrieve")
        self.add_edge("retrieve", "plan")
        self.add_edge("plan", "execute")
        self.add_edge("execute", "reflect")
        self.add_edge("reflect", END)

    def perceive(self, state: AgentState):
        pass

    def retrieve(self, state: AgentState):
        pass

    def plan(self, state: AgentState):
        pass

    def execute(self, state: AgentState):
        pass

    def reflect(self, state: AgentState):
        pass
    

if __name__ == "__main__":
    state = AgentState(name="John")

    graph = AgentWorkflow()
    compiled_graph = graph.compile()

    compiled_graph.get_graph().draw_mermaid_png(output_file_path="agent_workflow.png")
