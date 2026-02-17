import dspy


class IdentitySignature(dspy.Signature):
    """
    Write a concise description about the agent's personality, family situation and characteristics.
    """

    agent: str = dspy.InputField(desc="Name of the agent.")
    context: str = dspy.InputField(desc="Context about the agent.")

    identity: str = dspy.OutputField(desc="Concise identity description.")


def formulate_identity(agent: str, identity: str) -> str:
    # 'identity' arg here is actually the context in the original call
    try:
        predict = dspy.ChainOfThought(IdentitySignature)
        response = predict(agent=agent, context=identity)
        return response.identity
    except Exception as e:
        print(f"Error in formulate_identity: {e}")
        return f"{agent} is a person."


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.DummyLM([{"identity": "John Doe is..."}]))
    print(formulate_identity("John Doe", "John Doe is a 35 year old entrepreneur..."))
