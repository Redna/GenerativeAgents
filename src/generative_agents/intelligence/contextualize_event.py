import dspy


class ContextualizeEventSignature(dspy.Signature):
    """
    Write about the personality and observations of the agent based on a given event and related events.
    """

    agent: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Agent's identity context.")
    event_description: str = dspy.InputField(desc="Description of the perceived event.")
    events: str = dspy.InputField(desc="Related remembered events.")
    thoughts: str = dspy.InputField(desc="Agent's thoughts about the event.")

    event_context: str = dspy.OutputField(
        desc="Brief overview of things to remember for daily plan."
    )


def contextualize_event(
    agent: str, identity: str, event_description: str, events: str, thoughts: str
) -> str:
    try:
        predict = dspy.ChainOfThought(ContextualizeEventSignature)
        response = predict(
            agent=agent,
            identity=identity,
            event_description=event_description,
            events=events,
            thoughts=thoughts,
        )
        return response.event_context
    except Exception as e:
        print(f"Error in contextualize_event: {e}")
        return f"{agent} observed {event_description}."


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(
            lm=dspy.DummyLM([{"event_context": "John plans to study hard."}])
        )
    from pprint import pprint

    c = contextualize_event(
        agent="John",
        identity="John is a 22 year old student, who is learning a lot. He likes discussing with his peers. "
        "He is a social person.",
        event_description="John is going to the library to study for his exams.",
        events="The library is full of students. John is studying for his exams. John is a social person. "
        "John is a student. "
        "John is learning a lot. John likes discussing with his peers.",
        thoughts="John is a social person. The exam is important. He hopes to meet his friends at the library. "
        "He is excited to study.",
    )

    pprint(c)
