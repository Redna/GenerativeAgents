import dspy


class ActionLocationGameObjectSignature(dspy.Signature):
    """
    Identify the most relevant object for an action from the available objects.
    """

    action_description: str = dspy.InputField(desc="Current activity description.")
    available_objects: str = dspy.InputField(
        desc="Comma-separated list of available objects."
    )

    next_object: str = dspy.OutputField(
        desc="The most relevant object selected from the list."
    )


def action_location_game_object(action_description: str, available_objects: str) -> str:
    try:
        predict = dspy.Predict(ActionLocationGameObjectSignature)
        response = predict(
            action_description=action_description, available_objects=available_objects
        )

        allowed = [o.strip() for o in available_objects.split(",")]
        cleaned = response.next_object.strip()

        for o in allowed:
            if o.lower() == cleaned.lower():
                return o
        if allowed:
            return allowed[0]
        return ""
    except Exception as e:
        print(f"Error in action_location_game_object: {e}")
        return available_objects.split(",")[0] if available_objects else ""


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.DummyLM([{"next_object": "bed"}]))
    print(
        action_location_game_object(
            action_description="napping",
            available_objects="bed, easel, closet, painting",
        )
    )
