import dspy


class ActionPronunciatioSignature(dspy.Signature):
    """
    Provide one or two emoji that best represents the following statement or emotion.
    """

    action_description: str = dspy.InputField(desc="Statement or emotion description.")
    emoji: str = dspy.OutputField(desc="Maximum two emojis.")


def action_pronunciatio(action_description: str) -> str:
    try:
        predict = dspy.ChainOfThought(ActionPronunciatioSignature)
        response = predict(action_description=action_description)
        return response.emoji
    except Exception as e:
        print(f"Error in action_pronunciatio: {e}")
        return "😐"


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.DummyLM([{"emoji": "🚿"}]))
    print(action_pronunciatio(action_description="Taking a shower"))
    print(action_pronunciatio(action_description="Drinking"))
    print(action_pronunciatio(action_description="Taking a bath"))
    print(action_pronunciatio(action_description="Visiting a friend"))
    print(action_pronunciatio(action_description="Walking around"))
