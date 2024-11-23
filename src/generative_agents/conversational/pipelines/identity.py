from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="identity")

template = """Context:
{context}

Write a concise description about {agent}'s personality, family situation and characteristics. You include ALL the details provided in the given context (you MUST include all the names of persons, ages,...).

Identity: """

def formulate_identity(agent: str, previous_identity: str) -> str:
    content = template.format(agent=agent, context=previous_identity)
    identity = llm.invoke([HumanMessage(content=content)])

    return identity.content

if __name__ == "__main__":
    formulate_identity("John Doe", "John Doe is a 35 year old entrepreneur running his own start-up. He is dedicated to creating eco-friendly products. John is passionate about sustainability and environmental conservation. He practices yoga daily to stay focused and energized.")