from abc import ABC
from enum import Enum
from time import sleep
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client import models

from generative_agents.persistence.qdrant_wrapper import TimeAndImportanceWrapper, TimeAndImportanceBaseSchema

import sqlite3

_collections: Dict[str, TimeAndImportanceWrapper] = {}
_client = QdrantClient(":memory:")


class MemoryType(Enum):
    CHAT = "chat"
    OBSERVATION = "observation"
    EVENT = "event"
    THOUGHT = "thought"

class ConversationFilling(BaseModel):
    name: str
    utterance: str
    end: bool

class MemoryEntry(TimeAndImportanceBaseSchema):
    memory_type: str
    depth: int

    subject: str
    predicate: str
    object_: str

    poignancy: float = .5
    keywords: List[str] = []
    filling: List[str | ConversationFilling] = []

    hash_key: str = None

def initialize_agent(agent_name: str):
    if agent_name in _collections:
        raise Exception(f"Agent {agent_name} already exists")

    _collections[agent_name] = TimeAndImportanceWrapper(
        client=_client, collection_name=agent_name, data_schema=MemoryEntry)


def get(agent_name: str, context: str, limit=50) -> MemoryEntry:
    if not agent_name in _collections:
        raise Exception(f"Agent {agent_name} does not exist")

    result_set = _collections[agent_name].get_relevant_entries(context, limit=limit)
    return result_set

def get_by_hash(agent_name: str, hash_key:str):
    if not agent_name in _collections:
        raise Exception(f"Agent {agent_name} does not exist")

    collection = _collections[agent_name]

    filter = models.Filter(
        must=[
            models.FieldCondition(
                key="hash_key",
                match=models.MatchText(text=hash_key),
            )
        ]
    )

    result_set = collection.get_relevant_entries(query="", filter=filter)
    return result_set

def get_by_type(agent_name: str, context: str, memory_type: MemoryType):
    if not agent_name in _collections:
        raise Exception(f"Agent {agent_name} does not exist")

    collection = _collections[agent_name]

    filter = models.Filter(
        must=[
            models.FieldCondition(
                key="memory_type",
                match=models.MatchText(text=memory_type.value),
            )
        ]
    )

    result_set = collection.get_relevant_entries(context, filter=filter)
    return result_set

def get_last_chat(agent_name, with_agent_name) -> Optional[MemoryEntry]:
    if not agent_name in _collections:
        raise Exception(f"Agent {agent_name} does not exist")

    collection = _collections[agent_name]

    id = _get_last_conversation_id(agent_name, with_agent_name)

    return collection.get_by_id(id)

def get_active_chat(agent_name, with_agent_name) -> Optional[MemoryEntry]:
    if not agent_name in _collections:
        raise Exception(f"Agent {agent_name} does not exist")

    collection = _collections[agent_name]

    id = _get_active_conversation_id(agent_name, with_agent_name)
    record = collection.get_by_id(id)
    return MemoryEntry(**record.payload) if record else None

if __name__ == '__main__':

    initialize_database(True)
    initialize_agent("John Doe")

    add("John Doe", MemoryEntry(content="I am John Doe", memory_type=MemoryType.CHAT.value,
        depth=0, subject="John Doe", predicate="is", object="John Doe"))
    add("John Doe", MemoryEntry(content="The kitchen is dirty", memory_type=MemoryType.OBSERVATION.value,
        depth=0, subject="The kitchen", predicate="is", object="dirty"))

    add("John Doe", MemoryEntry(content="I love pizza", memory_type=MemoryType.CHAT.value,
        depth=0, subject="I", predicate="love", object="pizza"))
    sleep(1)
    first = add("John Doe", MemoryEntry(content="Talking with Mike Jones about the weather.",
                memory_type=MemoryType.CHAT.value, depth=0, subject="John Doe", predicate="is talking with", object="Mike Jones"))
    sleep(1)
    add("John Doe", MemoryEntry(content="The sky is blue", memory_type=MemoryType.OBSERVATION.value,
        depth=0, subject="The sky", predicate="is", object="blue"))
    sleep(1)
    add("John Doe", MemoryEntry(content="I am feeling happy today", memory_type=MemoryType.CHAT.value,
        depth=0, subject="I", predicate="am feeling", object="happy"))
    add("John Doe", MemoryEntry(content="The cat is sleeping", memory_type=MemoryType.OBSERVATION.value,
        depth=0, subject="The cat", predicate="is", object="sleeping"))
    add("John Doe", MemoryEntry(content="I am learning to code", memory_type=MemoryType.CHAT.value,
        depth=0, subject="I", predicate="am learning", object="to code"))
    sleep(1)
    last = add("John Doe", MemoryEntry(content="Talking with Mike Jones about Jeff barker.",
               memory_type=MemoryType.CHAT.value, depth=0, subject="John Doe", predicate="is talking with", object="Mike Jones"))
    entries = get("John Doe", "I am John Doe")
    print(entries)

    entries = get("John Doe", "I love pizza")

    entries = get_by_type("John Doe", "I am John Doe", MemoryType.CHAT)
    print(entries)

    entries = get_by_type(
        "John Doe", "Blue like the river", MemoryType.OBSERVATION)
    print(entries)

    entries = get_last_chat("John Doe", "Mike Jones")
    print(entries)

    add("John Doe", first)
    entries = get_last_chat("John Doe", "Mike Jones")
