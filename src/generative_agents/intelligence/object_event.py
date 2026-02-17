import dspy

class ObjectEventSignature(dspy.Signature):
    """
    Determine the state of an object that is being used by someone.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    object_name: str = dspy.InputField(desc="Name of the object.")
    action_description: str = dspy.InputField(desc="Description of the action being performed.")
    
    state: str = dspy.OutputField(desc="The new state of the object.")

def describe_object_state(name: str, object_name: str, object_address: str, action_description: str) -> tuple[str, tuple[str, str, str]]:
    try:
        predict = dspy.ChainOfThought(ObjectEventSignature)
        response = predict(
            name=name,
            object_name=object_name,
            action_description=action_description
        )
        return f"{object_name} is {response.state}", (object_address, "is", response.state)
    except Exception as e:
        print(f"Error in describe_object_state: {e}")
        return f"{object_name} is in use", (object_address, "is", "in use")

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "state": "clean"
         }]))
    print(describe_object_state(name="John Doe",
                                object_name="kitchen sink",
                                object_address="John Doe's house:kitchen sink",
                                action_description="washing dishes"))