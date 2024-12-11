import asyncio

import json
import os

from langgraph.constants import END, START, Send
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from langgraph.graph import StateGraph
from langfuse import callback
from dotenv import load_dotenv
from datetime import datetime
from generative_agents import global_state
from generative_agents.communication.models import AgentDTO, RoundUpdateDTO
from generative_agents.core.agent import Agent
from generative_agents.core.agent_runner import AgentRunner, AgentRunnerState
from generative_agents.core.memory.spatial import MemoryTree
from generative_agents.simulation.maze import BASE_PATH, Maze, Tile
from generative_agents.simulation.state import SimulationState
from generative_agents.simulation.time import SimulationTime
from generative_agents.utils import logger, add_log_level_for_agent

ROUND_UPDATE = "round_update"
REFLECT_CHANGES = "reflect_changes"

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


langfuse_handler = callback.CallbackHandler(
            secret_key=os.getenv("LANGFUSE_SK", ""),
            public_key=os.getenv("LANGFUSE_PK", ""),
            max_retries=3,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )

class RoundUpdateSnapshots:
    def __init__(self):
        self.rounds = []

    def add(self, time, agents: list[Agent]):
        agents_dto = [agent_runner.agent.to_dto() for agent_runner in agents.values()]

        converted_date_time = time.as_string()
        round_update = RoundUpdateDTO(
            round=len(self.rounds), time=converted_date_time, agents=agents_dto
        )
        self.rounds += [round_update]

    def get(self, round: int):
        return self.rounds[round]

    def get_all(self):
        return self.rounds

    @property
    def last(self):
        return self.rounds[-1]

    @property
    def current_round(self):
        return len(self.rounds)


class Simulation:
    def __init__(self, round_updates: RoundUpdateSnapshots, restore_from_step: int=None):
        self.maze = Maze()
        self.agents: dict[str, AgentRunner] = dict()
        self.__vision_start_tile = self.maze.get_random_tile()

        if restore_from_step:
            global_state.tick, global_state.time = self._restore_state_from_metadata(restore_from_step)

        # load the agents file
        with open(os.path.join(BASE_PATH, "agents/agent_backstory.json"), "r") as f:
            agents = json.load(f)["agents"]

        workflow = StateGraph(SimulationState)
        workflow.add_node(ROUND_UPDATE, self._round_update)
        workflow.add_node(REFLECT_CHANGES, self._reflect_changes)


        workflow.add_edge(START, ROUND_UPDATE)


        destinations = []
        self.initial_agent_states = {}
        for agent in agents:
            add_log_level_for_agent(agent["name"])
            agent_runner = self.initialize_agent(
                name=agent["name"],
                age=agent["age"],
                innate_traits=agent["innate_traits"],
                location=agent["location"],
                emoji=agent["emoji"],
                activity="idle",
                description=agent["description"],
            )

            self.agents[agent["name"]] = agent_runner

            workflow.add_node(agent["name"], agent_runner.workflow.compile())
            destinations.append(agent["name"])
            self.initial_agent_states[agent["name"]] = AgentRunnerState(
                daytype=None,
                agent_name=agent["name"],
                next_tile=agent["location"],
                perceived_events=[],
                retrieved=[],
                address=agent["location"],
                focused_event={},
                last_conversation=None,
            )

            workflow.add_edge(agent["name"], REFLECT_CHANGES)

            logger.log(agent["name"], f"Initialized {agent['name']}")


        workflow.add_conditional_edges(
            ROUND_UPDATE, self._distribute_state, destinations
        )

        workflow.add_edge(REFLECT_CHANGES, END)
        self.workflow = workflow

        self.round_updates = round_updates

    def stop(self):
        self.stopped = True

    async def run(self):

        load_dotenv()
        self.stopped = False

        compiled_graph = self.workflow.compile()

        print(compiled_graph.get_graph(xray=3).draw_mermaid())

        state = SimulationState(agent_states=self.initial_agent_states)


        current_time =  datetime.now().strftime("%Y%m%d%H%M%S")
        langfuse_session_id = "Run {current_time}"

        while not self.stopped:
            state = await compiled_graph.ainvoke(state,
                                                  subgraphs=False,
                                                  config={
                                                      "run_id": f"Round: {global_state.tick}, {global_state.time.as_string()} - {current_time}",
                                                      "callbacks": [langfuse_handler],
                                                      "metadata": {
                                                          "langfuse_session_id": langfuse_session_id
                                                      }
                                                    })


            if global_state.tick + 1 % 30 == 0:
                logger.info(f"tick: {global_state.tick}: Flushing langfuse queue")
                langfuse_handler.flush()


    def initialize_agent(
        self, name, age, innate_traits, location, emoji, activity, description
    ) -> AgentRunner:
        agent = Agent(
            name=name,
            age=age,
            time=global_state.time,
            innate_traits=innate_traits,
            location=location,
            emoji=emoji,
            activity=activity,
            tile=self.maze.address_tiles[location][-1],
            tree=self.initialize_visible_memory_tree(),
            description=description,
        )
        return AgentRunner(agent, self.agents, global_state.time, maze=self.maze)

    def initialize_visible_memory_tree(self):
        tree = MemoryTree()
        for tile in self.maze.get_nearby_tiles(self.__vision_start_tile, 1000):
            tree.add(tile)
        return tree

    def spawn_agent(self, data: AgentDTO):
        logger.log(data.name,
            f"spawning agent {data.name}, at {data.movement.col}, {data.movement.row}"
        )
        self.agents[data.name] = Agent.from_dto(data, self.maze, self.simulated_time)

    def _round_update(self, state: SimulationState) -> SimulationState:
        logger.info(
            f"round: {self.round_updates.current_round} time: {global_state.time.as_string()}"
        )
        return SimulationState(simulation_round=self.round_updates.current_round)

    def _distribute_state(self, state: SimulationState) -> SimulationState:
        agent_states = state.get("agent_states", {})
        maze = state.get("maze")

        return [Send(name, {"maze": maze, **agent_state}) for name, agent_state in agent_states.items()]

    def _reflect_changes(self, state: SimulationState) -> SimulationState:
        maze = state.get("maze")

        agent_states = state.get("agent_states", {})

        positions = []

        for name, agent_state in agent_states.items():
            next_tile = agent_state.get("next_tile", None)

            agent = self.agents[name].agent

            old_tile = agent.scratch.tile

            while agent.scratch.finished_action:
                action = agent.scratch.finished_action.pop(0)
                if action.event.subject in old_tile.events:
                    del old_tile.events[str(action.event.spo_summary)]

            action = agent.scratch.action
            event = action.event

            if action:
                if old_tile.get_unique_name != event.tile.get_unique_name:
                    next_tile.events[str(event.spo_summary)] = event
                    if str(event.spo_summary) in old_tile.events:
                        del old_tile.events[str(action.event.spo_summary)]

                if action.address in maze.address_tiles:
                    address_tiles = maze.address_tiles[action.address]

                    if any([tile for tile in address_tiles if tile == agent.scratch.tile]):
                        logger.log(agent.name, f"WARNING: {action.address} in the area of current tile")
                        next_tile.events[str(event.spo_summary)] = event

                object_action = action.object_action
                if object_action and object_action.event:
                    object_event = object_action.event
                    if object_action.address in maze.address_tiles:
                        address_tiles = maze.address_tiles[object_action.address]

                        if any([tile for tile in address_tiles if tile == agent.scratch.tile]):
                            logger.log(agent.name, f"WARNING: {object_action.address} in the area of current tile")
                            maze.address_tiles[object_action.address][0].events[
                                str(object_event.spo_summary)
                            ] = object_event
                    else:
                        logger.log(agent.name, f"WARNING: {object_action.address} not in maze")
            else:
                next_tile = agent.scratch.tile

            agent.scratch.tile = next_tile

            if old_tile != next_tile:
                logger.log(agent.name, f"{agent.scratch.name} moved from {old_tile} to {next_tile}")
            else:
                logger.log(agent.name, f"{agent.scratch.name} is still at {next_tile}")
            logger.log(agent.name, f"{agent.scratch.name} is {agent.emoji}")
            logger.log(agent.name, f"{agent.scratch.name} is {agent.scratch.description}")

            positions.append((name, agent.scratch.tile))
            agent.scratch.save()


        print("\n\n")
        print("-" * 80)
        maze.print_grid(positions=positions)
        print(f"round: {self.round_updates.current_round} time: {global_state.time.as_string()}")

        for name, agent_runner in self.agents.items():
            agent = agent_runner.agent
            initials = "".join([word[0] for word in name.split(" ")]).strip()
            print(f"{initials:<2}{agent.emoji:<2}: {agent.scratch.description:<50}")


        self.round_updates.add(global_state.time, self.agents)
        self._save_metadata()

        global_state.time.tick()
        return {"maze": maze}

    def _save_metadata():
        filename = str(global_state.tick).zfill(10)

        with open(f"./storage/world/{filename}") as fh:
            json.dump({
                "tick": global_state.tick,
                "time": global_state.SimulationTime.as_string(),
                "increment": global_state.SimulationTime.increment}, fh)

    def _restore_state_from_metadata(tick) -> tuple[int, SimulationTime]:
        filename = str(global_state.tick).zfill(10)

        with open(f"./storage/world/{filename}") as fh:
            restored = json.load(fh)
            tick = restored["tick"]
            simulation_time = SimulationTime(restored["increment"], restored["time"])

        return tick, simulation_time

async def main():
    round_updates = RoundUpdateSnapshots()
    simulation = Simulation(round_updates)
    # api.start(simulation.run_loop, simulation.spawn_agent)

    try:
        await simulation.run()
    except Exception:
        logger.exception("An error occurred")
        langfuse_handler.flush()


if __name__ == "__main__":
    asyncio.run(main())
