import datetime
from functools import lru_cache
import random

from haystack import component

from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.core.events import Event, EventType, PerceivedEvent
from generative_agents.core.whisper.whisper import whisper
from generative_agents.persistence.database import ConversationFilling

from generative_agents.conversational.pipelines.reflection_points import reflection_points
from generative_agents.conversational.pipelines.evidence_and_insights import evidence_and_insights
from generative_agents.conversational.pipelines.action_event_tripple import action_event_triple
from generative_agents.conversational.pipelines.memo_on_conversation import memo_on_conversation
from generative_agents.conversational.pipelines.planning_on_conversation import planning_on_conversation
from generative_agents.simulation.maze import Maze, Tile
from generative_agents.utils import timeit


@component
class Execution:
    def __init__(self, agent: 'Agent'):
        self.agent = agent

    @timeit
    @component.output_types(next_tile=Tile)
    def run(self, maze: Maze, agents: dict[str, 'Agent'], plan: str) -> Tile:
        if "<random>" in plan or self.agent.scratch.planned_path == []:
            self.agent.scratch.action_path_set = False

        # <action_path_set> is set to True if the path is set for the current action.
        # It is False otherwise, and means we need to construct a new path.
        if not self.agent.scratch.action_path_set:
            # <target_tiles> is a list of tile coordinates where the persona may go
            # to execute the current action. The goal is to pick one of them.
            target_tiles = None

            if "<persona>" in plan:
                # Executing persona-persona interaction.
                target_persona_tile = agents[plan.split(
                    "<persona>")[-1].strip()].scratch.tile
                potential_path = maze.find_path(self.agent.scratch.tile,
                                                target_persona_tile)

                if len(potential_path) <= 2:
                    target_tiles = [potential_path[0]]
                else:
                    potential_1 = maze.find_path(self.agent.scratch.tile,
                                                 potential_path[int(len(potential_path)/2)])
                    potential_2 = maze.find_path(self.agent.scratch.tile,
                                                 potential_path[int(len(potential_path)/2)+1])
                    if len(potential_1) <= len(potential_2):
                        target_tiles = [
                            potential_path[int(len(potential_path)/2)]]
                    else:
                        target_tiles = [
                            potential_path[int(len(potential_path)/2+1)]]

            elif "<waiting>" in plan:
                # Executing interaction where the persona has decided to wait before
                # executing their action.
                x = int(plan.split()[1])
                y = int(plan.split()[2])
                target_tiles = [[x, y]]

            elif "<random>" in plan:
                # Executing a random location action.
                target_tiles = [maze.get_random_tile(self.agent.scratch.tile)]
            else:
                # This is our default execution. We simply take the persona to the
                # location where the current action is taking place.
                # Retrieve the target addresses. Again, plan is an action address in its
                # string form. <maze.address_tiles> takes this and returns candidate
                # coordinates.
                if plan not in maze.address_tiles:
                    fallback_plan = ":".join(plan.split(":")[0:-1])

                    if fallback_plan not in maze.address_tiles:
                        fallback_plan = random.choice(
                            list(maze.address_tiles.keys()))

                    target_tiles = maze.address_tiles[fallback_plan]
                else:
                    target_tiles = maze.address_tiles[plan]

            # There are sometimes more than one tile returned from this (e.g., a tabe
            # may stretch many coordinates). So, we sample a few here. And from that
            # random sample, we will take the closest ones.
            if len(target_tiles) < 4:
                target_tiles = random.sample(
                    list(target_tiles), len(target_tiles))
            else:
                target_tiles = random.sample(list(target_tiles), 4)

            # If possible, we want personas to occupy different tiles when they are
            # headed to the same location on the maze. It is ok if they end up on the
            # same time, but we try to lower that probability.
            # We take care of that overlap here.
            persona_name_set = set(agent.name for agent in agents)
            new_target_tiles = []
            for tile in target_tiles:
                curr_event_set = tile.events
                pass_curr_tile = False
                for j in curr_event_set:
                    if j[0] in persona_name_set:
                        pass_curr_tile = True
                if not pass_curr_tile:
                    new_target_tiles += [tile]
            if len(new_target_tiles) == 0:
                new_target_tiles = target_tiles
            target_tiles = new_target_tiles
            # Now that we've identified the target tile, we find the shortest path to
            # one of the target tiles.
            curr_tile = self.agent.scratch.tile
            closest_target_tile = None
            path = None
            for i in target_tiles:
                # path_finder takes a collision_mze and the curr_tile coordinate as
                # an input, and returns a list of coordinate tuples that becomes the
                # path.
                # e.g., [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4)...]
                curr_path = maze.find_path(curr_tile, i)

                if not closest_target_tile:
                    closest_target_tile = i
                    path = curr_path
                elif len(curr_path) < len(path):
                    closest_target_tile = i
                    path = curr_path

            # Actually setting the <planned_path> and <action_path_set>. We cut the
            # first element in the planned_path because it includes the curr_tile.
            self.agent.scratch.planned_path = path[1:]
            self.agent.scratch.action_path_set = True

        # Setting up the next immediate step. We stay at our curr_tile if there is
        # no <planned_path> left, but otherwise, we go to the next tile in the path.
        ret = self.agent.scratch.tile
        if self.agent.scratch.planned_path:
            ret = self.agent.scratch.planned_path[0]
            self.agent.scratch.planned_path = self.agent.scratch.planned_path[1:]

        description = f"{self.agent.scratch.action.event.description}"
        description += f" @ {self.agent.scratch.action.address}"

        self.agent.emoji = self.agent.scratch.action.emoji
        self.agent.description = description
        return {"next_tile": ret}