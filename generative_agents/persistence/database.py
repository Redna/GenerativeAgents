import datetime
from enum import Enum
from time import sleep
from typing import List

from pydantic import BaseModel


from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from qdrant_client import models

from langchain.retrievers import TimeWeightedVectorStoreRetriever

from langchain.schema import Document
from qdrant_wrapper import TimeAndImportanceWrapper, TimeAndImportanceBaseSchema
from sentence_transformers import SentenceTransformer


retrievers = {}
client = QdrantClient(":memory:")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
VECTOR_PARAMS = models.VectorParams(
    size=embeddings.client.get_sentence_embedding_dimension(), distance=models.Distance.COSINE)
TOP_K = 20


class MemoryType(Enum):
    CHAT = "chat"
    OBSERVATION = "observation"
    EVENT = "event"
    THOUGHT = "thought"


class MemoryMetadataEntry(BaseModel):
    id: str | None = None
    memory_type: str
    depth: int
    created: datetime.datetime = datetime.datetime.now()
    expiration: datetime.datetime = None
    last_accessed: datetime.datetime = datetime.datetime.now()

    subject: str
    predicate: str
    object: str

    poignancy: float = .5
    keywords: List[str] = []
    filling: List[List[str]] = []


class MemoryEntry(BaseModel):
    description: str
    metadata: MemoryMetadataEntry

    @staticmethod
    def list_from_db(result_list) -> List['MemoryEntry']:
        return [MemoryEntry(description=entry.page_content, metadata=MemoryMetadataEntry(**entry.metadata)) for entry in result_list]

    def format_for_db(self):
        return [Document(page_content=self.description, metadata=self.metadata.dict())]


def initialize_agent(agent_name: str):
    if agent_name in retrievers:
        raise Exception(f"Agent {agent_name} already exists")

    client.create_collection(agent_name, vectors_config=VECTOR_PARAMS)
    client.create_payload_index(
        agent_name, field_name="metadata.memory_type", field_schema="keyword")
    client.create_payload_index(
        agent_name, field_name="metadata.created", field_schema="integer")
    vectorstore = Qdrant(
        client, collection_name=agent_name, embeddings=embeddings)

    retrievers[agent_name] = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, search_kwargs={"k": TOP_K}, other_score_keys=["poignancy"])


def add(agent_name: str, memory_entry: MemoryEntry) -> MemoryEntry:
    if agent_name not in retrievers:
        initialize_agent(agent_name)

    retriever = retrievers[agent_name]

    retriever.add_documents(memory_entry.format_for_db())

    return memory_entry


def get(agent_name: str, context: str) -> MemoryEntry:
    if not agent_name in retrievers:
        raise Exception(f"Agent {agent_name} does not exist")

    result_set = retrievers[agent_name].get_relevant_documents(context)
    print("retrieved: ", result_set)
    memories = MemoryEntry.list_from_db(result_set)

    return memories


def get_by_type(agent_name: str, context: str, memory_type: str):
    if not agent_name in retrievers:
        raise Exception(f"Agent {agent_name} does not exist")

    retriever: TimeWeightedVectorStoreRetriever = retrievers[agent_name]

    retriever.search_kwargs["filter"] = models.Filter(
        must=[
            models.FieldCondition(
                key="matadata.memory_type",
                match=models.MatchText(text=memory_type),
            )
        ]
    )

    result_set = retriever.get_relevant_documents(context)
    print("retrieved: ", result_set)
    memories = MemoryEntry.list_from_db(result_set)
    del retriever.search_kwargs["filter"]
    return memories


def get_recent_chat(agent_name, with_angent_name):
    if not agent_name in retrievers:
        raise Exception(f"Agent {agent_name} does not exist")

    documents = client.scroll(
        collection_name=agent_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="matadata.memory_type",
                    match=models.MatchText(text=MemoryType.CHAT.value),
                ),
                models.FieldCondition(
                    key="matadata.object",
                    match=models.MatchText(text=with_angent_name),
                ),
            ]
        )
    )

    return MemoryEntry.list_from_db(documents)

def main():
    initialize_agent("John Doe")
    add("John Doe", MemoryEntry(description="I am John Doe", metadata=MemoryMetadataEntry(
        memory_type="observation", subject="Richard Dean", predicate="is", object="lazy", depth=1)))
    add("John Doe", MemoryEntry(description="I am a lawyer", metadata=MemoryMetadataEntry(
        memory_type="observation", subject="John Doe", predicate="is", object="lawyer", depth=1)))
    add("John Doe", MemoryEntry(description="I am a butcher", metadata=MemoryMetadataEntry(
        memory_type="observation", subject="John Doe", predicate="is", object="butcher", depth=1)))
    add("John Doe", MemoryEntry(description="The weather is great", metadata=MemoryMetadataEntry(
        memory_type="observation", subject="The weather", predicate="is", object="great", depth=1)))

    add("John Doe", MemoryEntry(description="John Doe is having a converation with Jane Austin about her homework.",
                                metadata=MemoryMetadataEntry(memory_type="chat",
                                                             subject="John Doe",
                                                             predicate="chats with",
                                                             object="Jane Austin",
                                                             depth=1,
                                                             filling=[["Jane Austin", "Hey John, how are you?"],
                                                                      ["John Doe", "I am doing great, how about you?"],
                                                                      ["Jane Austin","I am doing great too!"],
                                                                      ["John Doe", "That's great to hear. How is your homework?!"]])))
    add("John Doe", MemoryEntry(description="John Doe is having a converation with Ralf Meyer about his vacation.",
                                metadata=MemoryMetadataEntry(memory_type="chat",
                                                             subject="John Doe",
                                                             predicate="chats with",
                                                             object="Ralf Meyer",
                                                             depth=1,
                                                             filling=[["Ralf Meyer", "Hey John, how are you?"],
                                                                      ["John Doe", "I am doing great, how about you?"],
                                                                      ["Ralf Meyer","Not too bad!"],
                                                                      ["John Doe", "How was your vacation?!"]])))
    
    add("John Doe", MemoryEntry(description="John Doe is having a converation with Jane Austin about her homework.",
                                metadata=MemoryMetadataEntry(memory_type="chat",
                                                             subject="John Doe",
                                                             predicate="chats with",
                                                             object="Jane Austin",
                                                             depth=1,
                                                             filling=[["Jane Austin", "Hey John, how are you?"],
                                                                      ["John Doe", "I am doing great, how about you?"],
                                                                      ["Jane Austin","I am doing great too!"],
                                                                      ["John Doe", "That's great to hear. How is your homework?!"]])))
    
    add("John Doe", MemoryEntry(description="John Doe is having a converation with Jane Austin about her homework.",
                                metadata=MemoryMetadataEntry(memory_type="chat",
                                                             subject="John Doe",
                                                             predicate="chats with",
                                                             object="Ralf Meyer",
                                                             depth=1,
                                                             filling=[["Jane Austin", "Hey John, how are you?"],
                                                                      ["John Doe", "I am doing great, how about you?"],
                                                                      ["Jane Austin","I am doing great too!"],
                                                                      ["John Doe", "That's great to hear. How is your homework?!"]])))

    result = get("John Doe", "Do you like meat?")

    related = get_by_type("John Doe", "John Doe is lazy", "observation")

    query_filter = get_by_type("John Doe", "Do you like meat?", "chat")

    chat = get_recent_chat("John Doe", "Jane Austin")
    print(result)


if __name__ == '__main__':    
    
    collection = TimeAndImportanceWrapper(client=client, collection_name="John Doe", embedding_model=model, data_schema=)
    
    collection.add([TimeAndImportanceBaseSchema(content="I am John Doe"), TimeAndImportanceBaseSchema(content="I am a lawyer"), TimeAndImportanceBaseSchema(content="I am a butcher"), TimeAndImportanceBaseSchema(content="The weather is great")])
    sleep(2)
    collection.add([TimeAndImportanceBaseSchema(content="The weather is great"), TimeAndImportanceBaseSchema(content="Once in a lifetime"), TimeAndImportanceBaseSchema(content="Time flies"), TimeAndImportanceBaseSchema(content="Flying to Europe")])
    
    sleep(2)
    
    entries = collection.get_relevant_entries("I am John Doe")
    print(entries)