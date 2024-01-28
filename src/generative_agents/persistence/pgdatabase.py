# good resources
# https://qdrant.tech/articles/hybrid-search/
# https://www.sbert.net/examples/applications/semantic-search/README.html

from abc import ABC
import asyncio
from dataclasses import asdict
from datetime import datetime
import itertools
from typing import List
from pgvector.psycopg import register_vector_async
import psycopg
from pgvector.sqlalchemy import Vector
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy.orm import Mapped, relationship


from sqlalchemy import ARRAY, JSON, Enum, Float, ForeignKey, create_engine, insert, select, text, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, mapped_column, Session

from generative_agents.core.events import PerceivedEvent

engine = create_engine('postgresql+psycopg://app:ml27ZXhcEZA5ZNgWd6dRxnmeJPW3EJpfq8Kt7KapZCqptUdnWyfkKE9cU8EOCMv7@192.168.178.111/app')

Base = declarative_base()

class MemoryType(Enum):
    CHAT = "chat"
    OBSERVATION = "observation"
    EVENT = "event"
    THOUGHT = "thought"

class ConversationFilling(Base):
    __tablename__ = 'conversation_filling'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: str
    utterance: str
    end: bool
    parent_id: Mapped[int] = mapped_column(ForeignKey("memory.id"))

class Memory(Base):
    __tablename__ = 'memory'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    expiration_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.now + datetime.timedelta(days=5))
    importance: Mapped[float] = mapped_column(Float)

    memory_type: Mapped[str] = mapped_column(Enum(MemoryType))
    depth: Mapped[int] = mapped_column(Integer)
    
    description: Mapped[str] = mapped_column(String)
    embeddings: Mapped[List[float]] = mapped_column(Vector(384))

    subject: Mapped[str] = mapped_column(String)
    predicate: Mapped[str] = mapped_column(String)
    object_: Mapped[str] = mapped_column(String)
    
    poignancy: Mapped[float] = mapped_column(Float)
    keywords: Mapped[List[str]] = mapped_column(ARRAY(String))
    filling: Mapped[List[ConversationFilling]] = relationship()

    hash_key: Mapped[str] = mapped_column(String)

    event: Mapped[dict] = mapped_column(JSON)

    def to_dataclass(self) -> PerceivedEvent:
        self.event[id] = self.id
        return PerceivedEvent(**self.event)

Base.metadata.create_all(engine)


sentences = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling',
    'The wolf is howling',
    'The lion is roaring',
    'The tiger is growling',
    'The bear is howling',
    'The bee is buzzing'
]
#query = 'growling bear'
query = 'What sound does a bear do?'


async def insert_data(conn):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    embeddings = model.encode(sentences)

    sql = 'INSERT INTO documents (content, embedding) VALUES ' + ', '.join(['(%s, %s)' for _ in embeddings])
    params = list(itertools.chain(*zip(sentences, embeddings)))
    await conn.execute(sql, params)


async def semantic_search(conn, query):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    embedding = model.encode(query)

    async with conn.cursor() as cur:
        await cur.execute('SELECT id, content FROM documents ORDER BY embedding <=> %s LIMIT 5', (embedding,))
        return await cur.fetchall()


async def keyword_search(conn, query):
    async with conn.cursor() as cur:
        await cur.execute("SELECT id, content FROM documents, plainto_tsquery('english', %s) query WHERE to_tsvector('english', content) @@ query ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC LIMIT 5", (query,))
        return await cur.fetchall()


def rerank(query, results):
    # deduplicate
    results = set(itertools.chain(*results))

    # re-rank
    encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = encoder.predict([(query, item[1]) for item in results])
    return [(v, s) for s, v in sorted(zip(scores, results), reverse=True)]


async def main():
    conn = await psycopg.AsyncConnection.connect(host="192.168.178.111", user="app", password="ml27ZXhcEZA5ZNgWd6dRxnmeJPW3EJpfq8Kt7KapZCqptUdnWyfkKE9cU8EOCMv7", dbname="app", autocommit=True)
    await create_schema(conn)
    await insert_data(conn)

    # perform queries in parallel
    results = await asyncio.gather(semantic_search(conn, query), keyword_search(conn, query))
    results_rerank = rerank(query, results)
    print(results)
    print(results_rerank)

if __name__ == '__main__':
    asyncio.run(main())
