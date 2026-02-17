import dspy

class ConversationSignature(dspy.Signature):
    """
    Generate the next utterance in a conversation and decide if it ends.
    """
    agent: str = dspy.InputField(desc="Name of the acting agent.")
    identity: str = dspy.InputField(desc="Identity of the acting agent.")
    memory: str = dspy.InputField(desc="Relevant memory of the agent.")
    past_context: str = dspy.InputField(desc="Context from past interactions.")
    location: str = dspy.InputField(desc="Current location.")
    agent_action: str = dspy.InputField(desc="Current action of the agent.")
    agent_with: str = dspy.InputField(desc="The person the agent is talking to.")
    agent_with_action: str = dspy.InputField(desc="Action of the other person.")
    conversation_history: str = dspy.InputField(desc="Conversation so far.")
    
    utterance: str = dspy.OutputField(desc="The next utterance.")
    end_conversation: bool = dspy.OutputField(desc="True if the conversation should end, False otherwise.")

def run_conversation(agent: str, identity: str, memory: str, past_context: str, location: str, agent_action: str, agent_with: str, agent_with_action: str, conversation: str) -> tuple[str, bool]:
    try:
        predict = dspy.ChainOfThought(ConversationSignature)
        response = predict(
            agent=agent,
            identity=identity,
            memory=memory,
            past_context=past_context,
            location=location,
            agent_action=agent_action,
            agent_with=agent_with,
            agent_with_action=agent_with_action,
            conversation_history=conversation
        )
        return response.utterance, response.end_conversation
    except Exception as e:
        print(f"Error in run_conversation: {e}")
        return "...", True

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "utterance": "Hello!",
             "end_conversation": False
         }]))

    print(run_conversation(identity="Alice Smith...",
                           memory="Alice Smith has known Emily...",
                           past_context="Alice Smith recommended...",
                           location="bookstore",
                           agent="Alice Smith",
                           agent_action="Searching...",
                           agent_with="Emily Johnson",
                           agent_with_action="Looking...",
                           conversation=""))
