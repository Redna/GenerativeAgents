from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You are {agent}. You will write about the personality and observations of {agent} based on a given event and related events.
Context
{identity}
{agent} perceived the following event: {event_description}
He remembered the following related events: {events}
He thought the following about the event: {thoughts}

What is {agent}'s personality and {agent}'s observations? Bring the event into context."""


class Context(BaseModel):
    event_context: str = Field(
        description="Contains a brief overview of things I need to remember to create my daily plan.")


def contextualize_event(agent: str, identity: str, event_description: str, events: str, thoughts: str) -> str:
    context = grammar_pipeline.run(model=Context, prompt_template=template, template_variables={
        "agent": agent,
        "identity": identity,
        "event_description": event_description,
        "events": events,
        "thoughts": thoughts
    })

    return context.event_context

if __name__ == "__main__":
    from pprint import pprint
    c = contextualize_event(agent="John", 
                        identity="John is a 22 year old student, who is learning a lot. He likes discussing with his peers. He is a social person.",
                        event_description="John is going to the library to study for his exams.",
                        events="The library is full of students. John is studying for his exams. John is a social person. John is a student. John is learning a lot. John likes discussing with his peers.",
                        thoughts="John is a social person. The exam is important. He hopes to meet his friends at the library. He is excited to study.")
    
    pprint(c)