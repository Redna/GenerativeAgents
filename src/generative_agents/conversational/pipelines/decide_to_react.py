from enum import Enum
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You will act as {{agent}}.

Context: {{context}}
Right now, it is {{current_time}}. 
{{agent}} is {{agent_observation}} when {{agent}} saw {{agent_with}} in the middle of {{agent_with_observation}}.",

Let's think step by step. Of the following two options, what should {{agent}} do?
- Option 1: Wait on {{initial_action_description}} until {{agent_with}} is done {{agent_with_action}}
- Option 2: Continue on to {{initial_action_description}} now
"""

class Options(Enum):
    Option1 = 1
    Option2 = 2

class DecideToReact(BaseModel):
    thought_process: str = Field(
        description="Contains the thought process of the agent to decide whether to react or not.")
    option: Options = Field(
        description="The option the agent decided to choose.")


def decide_to_react(context: str, current_time: str, agent: str, agent_with: str, agent_with_action: str, agent_observation: str, agent_with_observation: str, initial_action_description: str) -> int:
    decide_to_react = grammar_pipeline.run(model=DecideToReact, prompt_template=template, template_variables={
        "context": context,
        "current_time": current_time,
        "agent": agent,
        "agent_with": agent_with,
        "agent_with_action": agent_with_action,
        "agent_observation": agent_observation,
        "agent_with_observation": agent_with_observation,
        "initial_action_description": initial_action_description
    })

    return decide_to_react.option.value

if __name__ == "__main__":
    print(decide_to_react(context="You are walking through the park going to Johns Pub.",
                          current_time="10:00 PM",
                          agent="William Strange",
                          agent_with="John Smith",
                          agent_with_action="Jumping up and down",
                          agent_observation="walking through the park",
                          agent_with_observation="Jumping up and down in the park",
                          initial_action_description="walking through the park going to Johns Pub"))
    