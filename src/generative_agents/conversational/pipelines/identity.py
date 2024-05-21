from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """Your task is write a concise description about {{agent}}'s personality, family situation and characteristics. You include ALL the details provided in the given context (you MUST include all the names of persons, ages,...).
Respond in valid json format.

Context:
{{context}}

Task: Formulate the identity of {{agent}}."""

class Identity(BaseModel):
    identity: str = Field(description="Context of the agent's personality, family situation and characteristics.")

def formulate_identity(agent: str, identity: str) -> str:
    identity = grammar_pipeline.run(model=Identity, prompt_template=template, template_variables={
        "agent": agent,
        "context": identity
    })

    return identity.identity

if __name__ == "__main__":
    formulate_identity("John Doe", "John Doe is a 35 year old entrepreneur running his own start-up. He is dedicated to creating eco-friendly products. John is passionate about sustainability and environmental conservation. He practices yoga daily to stay focused and energized.")