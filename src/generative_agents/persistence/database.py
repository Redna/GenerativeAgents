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
_connection = sqlite3.connect('conversation.db')


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


def initialize_database(recreate: bool = False):
    if recreate:
        _connection.execute('DROP TABLE IF EXISTS active_conversations')
        _connection.execute('DROP TABLE IF EXISTS last_conversations')

    _connection.execute(
        'CREATE TABLE IF NOT EXISTS active_conversations (agent TEXT, with_agent TEXT, conversation_id TEXT, PRIMARY KEY (agent, with_agent))')
    _connection.execute(
        'CREATE TABLE IF NOT EXISTS last_conversations (agent TEXT, with_agent TEXT, conversation_id TEXT, PRIMARY KEY (agent, with_agent))')
    _connection.commit()


def _get_active_conversation_id(agent_name: str, with_agent_name: str):
    cursor = _connection.execute(
        'SELECT conversation_id FROM active_conversations WHERE agent = ? AND with_agent = ?', (agent_name, with_agent_name))
    result = cursor.fetchone()
    return result[-1] if result else None

def _get_last_conversation_id(agent_name: str, with_agent_name: str):
    cursor = _connection.execute(
        'SELECT conversation_id FROM last_conversations WHERE agent = ? AND with_agent = ?', (agent_name, with_agent_name))
    result = cursor.fetchone()
    return result[-1] if result else None

def _set_active_conversation_id(agent_name: str, with_agent_name: str, conversation_id: str):
    _connection.execute('INSERT OR REPLACE INTO active_conversations (agent, with_agent, conversation_id) VALUES (?, ?, ?)',
                        (agent_name, with_agent_name, conversation_id))
    _connection.commit()

def _set_last_conversation_id(agent_name: str, with_agent_name: str, conversation_id: str):
    _connection.execute('INSERT OR REPLACE INTO active_conversations (agent, with_agent, conversation_id) VALUES (?, ?, ?)',
                        (agent_name, with_agent_name, conversation_id))
    _connection.commit()

def _delete_active_conversation_id(agent_name: str, with_agent_name: str):
    _connection.execute(
        'DELETE FROM active_conversations WHERE agent = ? AND with_agent = ?', (agent_name, with_agent_name))
    _connection.commit()

def initialize_agent(agent_name: str):
    if agent_name in _collections:
        raise Exception(f"Agent {agent_name} already exists")

    _collections[agent_name] = TimeAndImportanceWrapper(
        client=_client, collection_name=agent_name, data_schema=MemoryEntry)


def add(agent_name: str, memory_entry: MemoryEntry) -> MemoryEntry:
    if agent_name not in _collections:
        initialize_agent(agent_name)

    collection = _collections[agent_name]
    memory_entry = collection.add([memory_entry])[0]

    if memory_entry.memory_type == MemoryType.CHAT.value:
        _set_active_conversation_id(
            agent_name, memory_entry.object_, memory_entry.id)
        
        if memory_entry.filling[-1] and memory_entry.filling[-1].end:
            _set_last_conversation_id(
                agent_name, memory_entry.object_, memory_entry.id)
            _delete_active_conversation_id(
                agent_name, memory_entry.object_)

    return memory_entry


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
