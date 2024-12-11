from langgraph.graph import StateGraph
from langgraph.constants import START, END

from generative_agents.core.agent import Agent
from generative_agents.core.cognitive_components.execution import Execution
from generative_agents.core.cognitive_components.perception import Perception
from generative_agents.core.cognitive_components.plan import Plan
from generative_agents.core.cognitive_components.reflection import Reflection
from generative_agents.core.cognitive_components.retrieval import Retrieval
from generative_agents.simulation.maze import Maze
from generative_agents.simulation.state import AgentRunnerState, SimulationState
from generative_agents.simulation.time import SimulationTime, DayType
from generative_agents.utils import logger

class AgentRunner:
    def __init__(self, agent: Agent, agents: dict[str, 'Agent'], time: SimulationTime, maze: Maze):
        self.agent = agent
        self.agents = agents
        self.time = time

        DAYTYPE_NODE = f"set_daytype - {agent.name}"
        PERCEPTION_NODE = f"perception - {agent.name}"
        RETRIEVAL_NODE = f"retrieval - {agent.name}"
        PLAN_NODE = f"plan - {agent.name}"
        EXECUTION_NODE = f"execution - {agent.name}"
        REFLECTION_NODE = f"reflection - {agent.name}"
        WRAP_UP = f"wrap_up - {agent.name}"

        workflow = StateGraph(AgentRunnerState, input=AgentRunnerState, output=SimulationState)

        workflow.add_node(DAYTYPE_NODE, self.set_daytype)
        workflow.add_node(PERCEPTION_NODE, Perception(self.agent, maze=maze).workflow.compile())
        workflow.add_node(RETRIEVAL_NODE, Retrieval(self.agent).workflow.compile())
        workflow.add_node(PLAN_NODE, Plan(self.agent, agents).workflow.compile())
        workflow.add_node(EXECUTION_NODE, Execution(self.agent, agents, maze=maze).workflow.compile())
        workflow.add_node(REFLECTION_NODE, Reflection(self.agent).workflow.compile())
        workflow.add_node(WRAP_UP, self.wrap_up)

        workflow.add_edge(START, DAYTYPE_NODE)
        workflow.add_edge(DAYTYPE_NODE, PERCEPTION_NODE)
        workflow.add_edge(PERCEPTION_NODE, RETRIEVAL_NODE)
        workflow.add_edge(RETRIEVAL_NODE, PLAN_NODE)
        workflow.add_edge(PLAN_NODE, EXECUTION_NODE)
        workflow.add_edge(EXECUTION_NODE, REFLECTION_NODE)
        workflow.add_edge(REFLECTION_NODE, WRAP_UP)
        workflow.add_edge(WRAP_UP, END)
        self.workflow = workflow

    def set_daytype(self, state: SimulationState) -> SimulationState:

        daytype: DayType = state.get("daytype")

        if not daytype:
            daytype = DayType.FIRST_DAY
        elif (self.agent.scratch.time.today != self.time.today):
            daytype = DayType.NEW_DAY
        else:
            daytype = DayType.SAME_DAY

        return AgentRunnerState(daytype=daytype)

    def wrap_up(self, state: AgentRunnerState) -> SimulationState:
        logger.log(state["agent_name"], "completed workflow")
        return SimulationState(agent_states={self.agent.name: state})
