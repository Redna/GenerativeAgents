import dspy

class MemoOnConversationSignature(dspy.Signature):
    """
    Write a memo on what the agent found interesting from the conversation.
    """
    agent: str = dspy.InputField(desc="Name of the agent.")
    conversation: str = dspy.InputField(desc="The conversation text.")
    
    memo: str = dspy.OutputField(desc="One sentence memo of what was interesting.")

def memo_on_conversation(agent: str, conversation: str) -> str:
    try:
        predict = dspy.ChainOfThought(MemoOnConversationSignature)
        response = predict(
            agent=agent,
            conversation=conversation
        )
        return response.memo
    except Exception as e:
        print(f"Error in memo_on_conversation: {e}")
        return "Nothing specific."

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "memo": "That was interesting."
         }]))
    print(memo_on_conversation(agent="Alice Smith", conversation="..."))