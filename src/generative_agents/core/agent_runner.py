from haystack import Pipeline

from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END

from generative_agents.core.agent import Agent
from generative_agents.core.cognitive_components.execution import Execution
from generative_agents.core.cognitive_components.perception import Perception
from generative_agents.core.cognitive_components.plan import Plan
from generative_agents.core.cognitive_components.reflection import Reflection
from generative_agents.core.cognitive_components.retrieval import Retrieval
from generative_agents.simulation.maze import Maze
from generative_agents.simulation.time import SimulationTime, DayType
from generative_agents.utils import timeit

class AgentRunnerState(TypedDict):
    pass

class AgentRunner:
    __TRACER_NAME = "tracer"

    def __init__(self, agent: Agent, maze: Maze, agents: dict[str, 'Agent']):
        self.agent = agent
        workflow = StateGraph(AgentRunnerState)
        workflow.add_node("perception", Perception(self.agent).workflow.compile())
        workflow.add_node("retrieval", Retrieval(self.agent).workflow.compile())
        workflow.add_node("plan", Plan(self.agent).workflow.compile())
        workflow.add_node("execution", Execution(self.agent).workflow.compile())
        workflow.add_node("reflection", Reflection(self.agent).workflow.compile())
        workflow.add_edge(START, "perception")
        workflow.add_edge("perception", "retrieval")
        workflow.add_edge("retrieval", "plan")
        workflow.add_edge("plan", "execution")
        workflow.add_edge("execution", "reflection")
        workflow.add_edge("reflection", END)
        self.workflow = workflow

    @timeit
    def update(self, time: SimulationTime, maze: Maze, agents: dict[str, 'Agent']):
        daytype: DayType = DayType.SAME_DAY

        if not self.agent.scratch.time:
            daytype = daytype.FIRST_DAY
        elif (self.agent.scratch.time.today != time.today):
            daytype = daytype.NEW_DAY

        self.agent.scratch.time = time

        perception = Perception(self.agent)
        retrieval = Retrieval(self.agent)
        plan = Plan(self.agent)
        execution = Execution(self.agent)
        reflection = Reflection(self.agent)


        agent_list = {agent.name: agent for agent in agents}

        perceived = perception.run(maze)["perceived_events"]
        retrieved = retrieval.run(perceived)["retrieved"]
        address = plan.run(agent_list, daytype, retrieved)["address"]
        next_tile = execution.run(maze, agent_list, address)["next_tile"]
        reflection.run()

        #result = self.pipeline.run(
        #            data={"perception": {"maze": maze},
        #                    "plan": {"agents": agents, "daytype": daytype},
        #                    "execution": {"maze": maze, "agents": agents}})

        return next_tile
