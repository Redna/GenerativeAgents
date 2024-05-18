from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You will act as a person in a role-playing game. You are in a conversation with another person. You will be given a context and a conversation so far. You need to output valid JSON describing the next utterance and whether the conversation ended with your utterance.
You are {agent}. Your identity is: 
{identity}

Here is the memory that is in {agent}'s head:
{memory}

Past Context:
{past_context}

Current Location: {location}

Current Context:
{agent} was {agent_action} when {agent} saw {agent_with} in the middle of {agent_with_action}.
{agent} is initiating a conversation with {agent_with}.

{agent} and {agent_with} are chatting. Here is their conversation so far:
{conversation}

Given the context above, what does {agent} say to {agent_with} next in the conversation? And did it end the conversation?"""


class ConversationRound(BaseModel):
    utterance: str = Field(
        description="The next utterance in the conversation.")
    end_conversation: bool = Field(
        description="Whether the conversation ended with the utterance.")


def run_conversation(agent: str, identity: str, memory: str, past_context: str, location: str, agent_action: str, agent_with: str, agent_with_action: str, conversation: str) -> str:
    conversation_round = grammar_pipeline.run(model=ConversationRound, prompt_template=template, template_variables={
        "agent": agent,
        "identity": identity,
        "memory": memory,
        "past_context": past_context,
        "location": location,
        "agent_action": agent_action,
        "agent_with": agent_with,
        "agent_with_action": agent_with_action,
        "conversation": conversation
    })

    return conversation_round.utterance, conversation_round.end_conversation

if __name__ == "__main__":

    print(run_conversation(identity="Alice Smith is a librarian at the City Library. She is appreciated for her helpful nature.",
                           memory="Alice Smith has known Emily Johnson for 3 years. They met at a book club.",
                           past_context="Alice Smith recommended a mystery novel to Emily Johnson last month.",
                           location="bookstore",
                           agent="Alice Smith",
                           agent_action="Searching for new books for the library.",
                           agent_with="Emily Johnson",
                           agent_with_action="Looking for a birthday gift for her nephew.",
                           conversation=""))

    print(run_conversation(identity="Mark Turner is a gym instructor at FitLife Gym. He is popular for his motivational coaching.",
                           memory="Mark Turner has been friends with David Lee for 2 years. They met during a fitness workshop.",
                           past_context="Mark Turner was helping David Lee with a new workout routine yesterday.",
                           location="sports store",
                           agent="Mark Turner",
                           agent_action="Buying new gym equipment for FitLife Gym.",
                           agent_with="David Lee",
                           agent_with_action="Choosing running shoes for his marathon training.",
                           conversation=""))

    print(run_conversation(identity="Sarah Johnson is a chef at The Gourmet Kitchen. She is famous for her innovative recipes.",
                           memory="Sarah Johnson knows Chloe Adams for 6 months. They met at a cooking class.",
                           past_context="Sarah Johnson was teaching Chloe Adams how to bake French pastries last Saturday.",
                           location="farmer's market",
                           agent="Sarah Johnson",
                           agent_action="Selecting fresh produce for her restaurant.",
                           agent_with="Chloe Adams",
                           agent_with_action="Buying organic herbs for her home garden.",
                           conversation=""))
