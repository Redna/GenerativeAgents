from functools import lru_cache

import dspy


class RatePoignance(dspy.Signature):
    """Rate the poignance (importance) of an event on a scale of 0 to 10."""

    agent_name: str = dspy.InputField(desc="The name of the agent.")
    agent_identity: str = dspy.InputField(
        desc="A description of the agent's identity and backstory."
    )
    event_type: str = dspy.InputField(
        desc="The type of the event (e.g., 'Exhibition', 'Chat')."
    )
    description: str = dspy.InputField(desc="A description of the event.")
    rating: int = dspy.OutputField(
        desc="The rating of the poignance of the event (0-10)."
    )


@lru_cache(maxsize=2048)
def rate_poignance(
    agent_name: str, agent_identity: str, type_: str, description: str
) -> int:
    try:
        # Using ChainOfThought for better reasoning, though Predict is faster
        predict = dspy.ChainOfThought(RatePoignance)
        response = predict(
            agent_name=agent_name,
            agent_identity=agent_identity,
            event_type=type_,
            description=description,
        )
        # Ensure rating is within bounds
        return max(0, min(10, response.rating))
    except Exception as e:
        print(f"Error in rate_poignance: {e}")
        return 5  # Default safe value


def __tests():
    # Setup dummy dspy for tests if not configured
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.DummyLM([{"rating": 8}]))

    print(
        rate_poignance(
            "Emily Tan",
            "Emily Tan is a renowned sculptor...",
            "Exhibition",
            "Emily Tan is preparing for her solo art exhibition.",
        )
    )


if __name__ == "__main__":
    __tests()
