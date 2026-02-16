import dspy

class DecideToTalkSignature(dspy.Signature):
    """
    Decide whether to initiate a conversation based on the context, current time, and observations.
    """
    context: str = dspy.InputField(desc="The context of the situation.")
    current_time: str = dspy.InputField(desc="The current time.")
    init_agent: str = dspy.InputField(desc="The name of the agent deciding to talk.")
    agent_with: str = dspy.InputField(desc="The name of the agent to potentially talk to.")
    last_chat_summary: str = dspy.InputField(desc="Summary of the last conversation between the agents.")
    init_agent_observation: str = dspy.InputField(desc="What the initiating agent is currently doing.")
    agent_with_observation: str = dspy.InputField(desc="What the other agent is currently doing.")
    
    thought_process: str = dspy.OutputField(desc="The reasoning behind the decision.")
    initiate_conversation: bool = dspy.OutputField(desc="True if the agent decides to initiate a conversation, False otherwise.")

def decide_to_talk(context: str, current_time: str, init_agent: str, agent_with: str, last_chat_summary: str, init_agent_observation: str, agent_with_observation: str) -> bool:
    try:
        predict = dspy.ChainOfThought(DecideToTalkSignature)
        response = predict(
            context=context,
            current_time=current_time,
            init_agent=init_agent,
            agent_with=agent_with,
            last_chat_summary=last_chat_summary,
            init_agent_observation=init_agent_observation,
            agent_with_observation=agent_with_observation
        )
        return response.initiate_conversation
    except Exception as e:
        print(f"Error in decide_to_talk: {e}")
        return False

if __name__ == "__main__":
    # Setup dummy dspy for tests if not configured
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{"thought_process": "It seems appropriate.", "initiate_conversation": True}]))

    from pprint import pprint

    pprint(decide_to_talk(context="You are in the supermarket...",
                          current_time="5:00 PM",
                          init_agent="Jaiden Smith",
                          agent_with="John Doe",
                          last_chat_summary="last chatted a month ago about a movie.",
                          init_agent_observation="reading the label on a cereal box.",
                          agent_with_observation="deep in thought, looking at different kinds of tea."))
