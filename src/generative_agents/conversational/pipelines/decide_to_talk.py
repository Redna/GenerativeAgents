from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You will act as {{init_agent}}. Based on a given scenario you whether to initiate a conversation or not. You will only generate exactly 1 valid JSON.
Context: {{context}}

Right now, it is {{current_time}}. {{init_agent}} and {{agent_with}} {{last_chat_summary}}.
{{init_agent}} is currently {{init_agent_observation}}
{{agent_with_observation}}

Would {{init_agent}} initiate a conversation with {{agent_with}}? Think it through."""


class DecideToTalk(BaseModel):
    thought_process: str = Field(
        description="Contains the thought process of the agent to decide whether to initiate a conversation or not.")
    initiate_conversation: bool = Field(
        description="Whether the agent decided to initiate a conversation or not.")


def decide_to_talk(context: str, current_time: str, init_agent: str, agent_with: str, last_chat_summary: str, init_agent_observation: str, agent_with_observation: str) -> bool:
    decide_to_talk = grammar_pipeline.run(model=DecideToTalk, prompt_template=template, template_variables={
        "context": context,
        "current_time": current_time,
        "init_agent": init_agent,
        "agent_with": agent_with,
        "last_chat_summary": last_chat_summary,
        "init_agent_observation": init_agent_observation,
        "agent_with_observation": agent_with_observation
    })

    return decide_to_talk.initiate_conversation

if __name__ == "__main__":
    from pprint import pprint

    pprint(decide_to_talk(context="You are in the supermarket. Buying some groceries. You see your friend, John, in the same aisle as you.",
                          current_time="5:00 PM",
                          init_agent="Jaiden Smith",
                          agent_with="John Doe",
                          last_chat_summary="last chatted a month ago about a movie.",
                          init_agent_observation="reading the label on a cereal box.",
                          agent_with_observation="deep in thought, looking at different kinds of tea."))
    
    pprint(decide_to_talk(context="You are in the supermarket. Buying some groceries. You see your friend, John, in the same aisle as you.",
                          current_time="5:00 PM",
                          init_agent="Jaiden Smith",
                          agent_with="John Doe",
                          last_chat_summary="never chatted before.",
                          init_agent_observation="looking for a new brand of coffee.",
                          agent_with_observation="talking on the phone, seems engaged in the conversation."))
    
    pprint(decide_to_talk(context="You are in the supermarket. Buying some groceries. You see your friend, John, in the same aisle as you.",
                            current_time="5:00 PM",
                            init_agent="Jaiden Smith",
                            agent_with="John Doe",
                            last_chat_summary="last chatted a month ago about a movie.",
                            init_agent_observation="reading the label on a cereal box.",
                            agent_with_observation="deep in thought, looking at different kinds of tea."))
