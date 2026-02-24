import datetime
import unittest
from unittest.mock import MagicMock

from generative_agents.common.events import Event
from generative_agents.common.percept import Percept
from generative_agents.simulation.engine import SimulationEngine

# from generative_agents.core.agent import Agent # Avoid importing Agent to skip huge dependencies
from generative_agents.simulation.maze import Maze, Tile
from generative_agents.simulation.time import SimulationTime


# Mock Agent for testing
class MockAgent:
    def __init__(self, name):
        self.name = name
        self.working_memory = MagicMock()
        self.working_memory.tile = MagicMock()
        self.working_memory.finished_actions = []
        self.working_memory.action = MagicMock()
        self.working_memory.action.object_action = None
        self.run_step = MagicMock()

    def to_dto(self):
        # Return a dict that satisfies AgentDTO structure (or a mock that passes pydantic validation
        # if lenient, but dict is safer)
        # Actually Pydantic validates against the model.
        # Let's import AgentDTO if possible or mock the data structure.
        # Minimal fields for AgentDTO: name, age, inniate_traits, description, location, emoji, activity, movement
        return {
            "name": self.name,
            "age": 25,
            "inniate_traits": [],
            "description": "Test Description",
            "location": "World:Sector:Arena:Object",
            "emoji": "🧪",
            "activity": "testing",
            "movement": {"col": 10, "row": 10},
        }


class TestSimulationEngine(unittest.TestCase):
    def setUp(self):
        # Mock Maze
        self.maze = MagicMock(spec=Maze)
        self.maze.get_nearby_tiles.return_value = []
        self.maze.address_tiles = {}

        # Mock Time
        self.time = MagicMock(spec=SimulationTime)
        self.time.today = "Monday"
        self.time.time = datetime.datetime.now()
        self.time.as_string.return_value = "2023-01-01T00:00:00"  # Fixed return value

        # Mock Agent
        self.agent = MockAgent("TestAgent")
        self.agent.working_memory.tile = MagicMock()  # Removed spec=Tile
        self.agent.working_memory.tile.events = {}  # Initialize events
        self.agent.working_memory.tile.x = 10
        self.agent.working_memory.tile.y = 10
        self.agent.working_memory.tile.get_path.return_value = "World:Sector:Arena"
        self.agent.working_memory.vision_radius = 5
        self.agent.working_memory.attention_bandwidth = 10
        self.agent.working_memory.finished_actions = []
        self.agent.working_memory.action.event.subject = "TestAgent"
        self.agent.working_memory.action.object_action = None

        # Setup Engine
        self.engine = SimulationEngine(self.maze, [self.agent], self.time)

    def test_get_percept(self):
        # Setup mock tiles with events
        tile_with_event = MagicMock(spec=Tile)
        tile_with_event.x = 11
        tile_with_event.y = 10
        tile_with_event.get_path.return_value = "World:Sector:Arena"

        event = MagicMock(spec=Event)
        event.spo_summary = ("Subject", "Predicate", "Object")
        event.dist = 1
        event.entity_id = "Subject"
        tile_with_event.events = {"Subject": event}

        self.maze.get_nearby_tiles.return_value = [tile_with_event]

        # Call get_percept
        percept = self.engine.get_percept(self.agent)

        self.assertIsInstance(percept, Percept)
        self.assertEqual(len(percept.nearby_tiles), 1)
        self.assertEqual(len(percept.events), 1)
        self.assertEqual(percept.events[0], event)

    def test_step_calls_agent_run_step(self):
        # Mock calculate_percepts to avoid complex logic
        self.engine._calculate_percepts = MagicMock(
            return_value={"TestAgent": Percept()}
        )

        self.engine.step()

        self.agent.run_step.assert_called_once()
        self.engine._calculate_percepts.assert_called_once()


if __name__ == "__main__":
    unittest.main()
