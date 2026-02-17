import dspy


class ConversationSummarySignature(dspy.Signature):
    """
    Summarize a conversation in one sentence.
    """

    conversation: str = dspy.InputField(desc="The conversation text.")
    summary: str = dspy.OutputField(desc="One sentence summary.")


def conversation_summary(conversation: str) -> str:
    try:
        predict = dspy.ChainOfThought(ConversationSummarySignature)
        response = predict(conversation=conversation)
        return response.summary
    except Exception as e:
        print(f"Error in conversation_summary: {e}")
        return "Conversation happened."


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(
            lm=dspy.DummyLM([{"summary": "They greeted each other."}])
        )
    print(
        conversation_summary(
            conversation="Rudolf: Hello, how are you?\nJoanne: I am fine, thank you."
        )
    )
    print(
        conversation_summary(
            conversation="""Joe Walther: Hello, did you hear about Jim's party?
Frodo Reimsi: No, tell me more. You mean Jimmy Fraser?
Joe Walther: Jim Knofi. He is giving a dinner party.
Frodo Reimsi: I did not know that."""
        )
    )
