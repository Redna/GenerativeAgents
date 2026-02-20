import shutil
import pytest
import os
from datetime import datetime
import generative_agents.persistence.database
from generative_agents.persistence.database import MemoryEntry, MemoryType, get_repository
from generative_agents.agents.memory.repository import QdrantMemoryRepository


@pytest.fixture
def repo(tmp_path):
    """Sets up a fresh QdrantMemoryRepository backed by a local tmp directory."""
    # Clear global registry
    generative_agents.persistence.database._repositories = {}

    test_dir = str(tmp_path / "test_data")
    repository = QdrantMemoryRepository("TestAgent", data_dir=test_dir)
    generative_agents.persistence.database._repositories["TestAgent"] = repository

    yield repository

    generative_agents.persistence.database._repositories.clear()


def test_qdrant_persistence(repo, tmp_path):
    """Verifies: add → ANN retrieve → graph edge add → graph edge retrieve → restart (new repo same path)."""

    # 1. Add memories
    entry = MemoryEntry(
        id="mem1",
        content="Test memory persistence",
        created_at=datetime.now(),
        last_accessed_at=datetime.now(),
        memory_type=MemoryType.THOUGHT.value,
        depth=1,
        subject="Test",
        predicate="is",
        object_="Persistent",
    )
    generative_agents.persistence.database.add("TestAgent", entry)

    entry2 = MemoryEntry(
        id="mem2",
        content="Child memory derived from test",
        created_at=datetime.now(),
        last_accessed_at=datetime.now(),
        memory_type=MemoryType.THOUGHT.value,
        depth=1,
        subject="Child",
        predicate="is",
        object_="Derived",
    )
    generative_agents.persistence.database.add("TestAgent", entry2)

    # 2. Add graph edge
    generative_agents.persistence.database.add_event_link(entry.id, entry2.id, "caused")

    # 3. ANN retrieval works
    results = generative_agents.persistence.database.get("TestAgent", "Test memory persistence")
    assert len(results) > 0
    assert results[0].content == "Test memory persistence"

    # 4. Graph edge retrieval works
    related = generative_agents.persistence.database.get_related_events(entry.id, "caused")
    assert len(related) > 0
    assert related[0][0] == entry2.id

    # 5. Simulate restart — close the first client, open a new one at the same path
    test_dir = str(tmp_path / "test_data")
    repo._client.close()  # release the file lock before re-opening
    generative_agents.persistence.database._repositories.clear()

    new_repo = QdrantMemoryRepository("TestAgent", data_dir=test_dir)
    generative_agents.persistence.database._repositories["TestAgent"] = new_repo

    # ANN retrieval still works after restart
    results_after = generative_agents.persistence.database.get("TestAgent", "Test memory persistence")
    assert len(results_after) > 0, "Memories should persist across repo restarts"
    assert results_after[0].content == "Test memory persistence"

    # Graph edges persist in Qdrant payload
    related_after = generative_agents.persistence.database.get_related_events(entry.id, "caused")
    assert len(related_after) > 0, "Graph edges should persist in Qdrant payload"
    assert related_after[0][0] == entry2.id
