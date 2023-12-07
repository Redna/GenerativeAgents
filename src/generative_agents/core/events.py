
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
from generative_agents import global_state

from generative_agents.persistence.database import ConversationFilling, MemoryEntry
from pydantic import BaseModel


class EventType(Enum):
    EVENT = "event"
    THOUGHT = "thought"
    CHAT = "chat"
    OBSERVATION = "observation"
    PLAN = "plan"


@dataclass
class Event:
    depth: int
    subject: str
    predicate: str
    object_: str
    description: str
    filling: List[ConversationFilling | str] = field(default_factory=list)

    @property
    def spo_summary(self):
        return (self.subject, self.predicate, self.object_)


@dataclass
class PerceivedEvent(Event):
    id: str = None
    event_type: EventType = EventType.EVENT
    poignancy: float = .5
    created: datetime.datetime = global_state.time.time
    expiration: datetime.datetime = None
    last_accessed: datetime.datetime = global_state.time.time
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_db_entry(cls, entry: MemoryEntry):
        return cls(id=entry.id,
                   depth=entry.depth,
                   subject=entry.subject,
                   predicate=entry.predicate,
                   object_=entry.object_,
                   description=entry.content,
                   event_type=EventType(entry.memory_type),
                   poignancy=entry.poignancy,
                   created=entry.created_at,
                   expiration=entry.expiration_date,
                   last_accessed=entry.last_accessed_at,
                   filling=entry.filling,
                   keywords=entry.keywords)

    def to_db_entry(self):
        return MemoryEntry(
                           content=self.description,
                           memory_type=self.event_type.value,
                           depth=self.depth,
                           created_at=self.created,
                           expiration_date=self.expiration,
                           last_accessed_at=self.last_accessed,
                           subject=self.subject,
                           predicate=self.predicate,
                           object_=self.object_,
                           poignancy=self.poignancy,
                           keywords=self.keywords,
                           filling=self.filling)


class ObjectAction(BaseModel):
    address: str
    emoji: str
    event: Event


class Action(BaseModel):
    address: str
    start_time: datetime.datetime
    duration: int
    emoji: str
    event: Event
    object_action: ObjectAction = None

    @classmethod
    def idle(cls, address: str):
        return cls(address=address,
                   start_time=global_state.time.time,
                   duration=0,
                   emoji="⏳",
                   event=Event(depth=0, subject="", predicate="", object_="", description="idle"))
