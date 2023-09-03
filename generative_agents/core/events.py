
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from core.memory import MemoryEntry, MemoryMetadataEntry
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

    @property
    def spo_summary(self):
        return (self.subject, self.predicate, self.object_)
    


@dataclass
class PerceivedEvent(Event):
    id: str = None
    event_type: EventType
    poignancy: float = .5
    created: datetime.datetime = datetime.datetime.now()
    expiration: datetime.datetime = None
    last_accessed: datetime.datetime = datetime.datetime.now()
    filling: List[str] = []
    keywords: List[str] = []
    
    @classmethod
    def from_db_entry(cls, text: str, metadata: Dict[str, any]):
        return cls(id=metadata["id"],
                    type=metadata["memory_type"],
                    depth=metadata["depth"],
                    created=metadata["created"],
                    expiration=metadata["expiration"],
                    last_accessed=metadata["last_accessed"],
                    subject=metadata["subject"],
                    predicate=metadata["predicate"],
                    object_=metadata["object_"],
                    description=text,
                    poignancy=metadata["poignancy"],
                    keywords=metadata["keywords"],
                    filling=metadata["filling"])

    def to_db_entry(self):
        metadata = MemoryMetadataEntry(id=self.id,
                                       memory_type=self.type.value,
                                       depth=self.depth,
                                       created=self.created,
                                       expiration=self.expiration,
                                       last_accessed=self.last_accessed,
                                       subject=self.subject,
                                       predicate=self.predicate,
                                       object_=self.object_,
                                       poignancy=self.poignancy,
                                       keywords=self.keywords,
                                       filling=self.filling)

        return MemoryEntry(text=self.description,
                           metadata=metadata)

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