import dspy

class PlanningOnConversationSignature(dspy.Signature):
    """
    Determine what to remember from a conversation (in first person).
    """
    agent: str = dspy.InputField(desc="Name of the agent.")
    conversation: str = dspy.InputField(desc="The conversation text.")
    
    to_remember: str = dspy.OutputField(desc="One sentence on what to remember.")

def planning_on_conversation(agent: str, conversation: str) -> str:
    try:
        predict = dspy.ChainOfThought(PlanningOnConversationSignature)
        response = predict(
            agent=agent,
            conversation=conversation
        )
        return response.to_remember
    except Exception as e:
        print(f"Error in planning_on_conversation: {e}")
        return "I had a conversation."

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "to_remember": "I need to buy milk."
         }]))
    planning_on_conversation(agent="Alice Smith", conversation="...")
