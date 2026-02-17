import shutil
import pytest
import os
import json
from datetime import datetime
import generative_agents.persistence.database
from generative_agents.persistence.database import MemoryEntry, MemoryType, get_repository
from generative_agents.agents.memory.repository import JSONMemoryRepository

# Fixture to handle setup and teardown
@pytest.fixture
def repo(tmp_path):
    # Setup
    test_dir = tmp_path / "test_data_json"
    str_test_dir = str(test_dir)
    
    # Mock the repo registry
    generative_agents.persistence.database._repositories = {}
    
    # Initialize repository with test path
    repository = JSONMemoryRepository("TestAgent", data_dir=str_test_dir)
    generative_agents.persistence.database._repositories["TestAgent"] = repository
    
    yield repository
    
    # Teardown (optional with tmp_path as it cleans up itself, but good for explicit registry clearing)
    generative_agents.persistence.database._repositories.clear()

def test_json_persistence(repo, tmp_path):
    # 1. Add memory
    entry = MemoryEntry(
        id="mem1",
        content="Test memory persistence",
        created_at=datetime.now(),
        last_accessed_at=datetime.now(),
        memory_type=MemoryType.THOUGHT.value,
        depth=1,
        subject="Test",
        predicate="is",
        object_="Persistent"
    )
    generative_agents.persistence.database.add("TestAgent", entry)
    
    # Verify file exists
    # repo.base_dir is constructed from data_dir/AgentName
    expected_path = os.path.join(repo.base_dir, "memory.json")
    assert os.path.exists(expected_path)
    
    # 2. Add Graph Edge
    entry2 = MemoryEntry(
        id="mem2",
        content="Child memory",
        created_at=datetime.now(),
        last_accessed_at=datetime.now(),
        memory_type=MemoryType.THOUGHT.value,
        depth=1,
        subject="Child",
        predicate="is",
        object_="Derived"
    )
    generative_agents.persistence.database.add("TestAgent", entry2)
    
    generative_agents.persistence.database.add_event_link(entry.id, entry2.id, "caused")
    
    # 3. Simulate Restart
    # Clear registry
    generative_agents.persistence.database._repositories.clear()
    
    # Re-init explicitly with same path to simulate reload
    new_repo = JSONMemoryRepository("TestAgent", data_dir=repo.base_dir.replace("/TestAgent", ""))
    generative_agents.persistence.database._repositories["TestAgent"] = new_repo
    
    # 4. Verify Memory Exists via Retrieval
    results = generative_agents.persistence.database.get("TestAgent", "Test memory persistence")
    assert len(results) > 0
    assert results[0].content == "Test memory persistence"
    
    # 5. Verify Graph Edge Exists
    related = generative_agents.persistence.database.get_related_events(entry.id, "caused")
    assert len(related) > 0
    assert related[0][0] == entry2.id
