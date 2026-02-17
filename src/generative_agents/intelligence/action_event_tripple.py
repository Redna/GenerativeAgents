import dspy


class ActionEventTripleSignature(dspy.Signature):
    """
    Given a sentence identify the subject, predicate, and object from the sentence.
    """

    name: str = dspy.InputField(desc="Name of the agent.")
    action_description: str = dspy.InputField(desc="Description of action.")

    subject: str = dspy.OutputField(
        desc="The subject of the sentence (usually the name)."
    )
    predicate: str = dspy.OutputField(desc="The action being performed.")
    object: str = dspy.OutputField(
        desc="The entity that the action is being performed on."
    )


def action_event_triple(
    name: str, action_description: str, address: str = None
) -> tuple[str, str, str]:
    try:
        predict = dspy.ChainOfThought(ActionEventTripleSignature)
        response = predict(name=name, action_description=action_description)

        subject = response.subject
        if address:
            subject = address

        return (subject, response.predicate, response.object)
    except Exception as e:
        print(f"Error in action_event_triple: {e}")
        # Fallback
        return (address if address else name, "is", "doing something")


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(
            lm=dspy.DummyLM(
                [{"subject": "John Doe", "predicate": "taking", "object": "shower"}]
            )
        )
    print(
        action_event_triple(
            name="John Doe", action_description="John Doe is taking a warm shower"
        )
    )
