import dspy
from enum import Enum

class Options(Enum):
    Option1 = 1
    Option2 = 2

class DecideToReactSignature(dspy.Signature):
    """
    Decide whether to react to an observation or wait, based on the context.
    Option 1: Wait.
    Option 2: Continue.
    """
    context: str = dspy.InputField(desc="The context of the situation.")
    current_time: str = dspy.InputField(desc="The current time.")
    agent: str = dspy.InputField(desc="The name of the agent deciding.")
    agent_with: str = dspy.InputField(desc="The name of the other agent involved.")
    agent_with_action: str = dspy.InputField(desc="What the other agent is doing.")
    agent_observation: str = dspy.InputField(desc="What the deciding agent is currently doing.")
    agent_with_observation: str = dspy.InputField(desc="Observation of the other agent.")
    initial_action_description: str = dspy.InputField(desc="The initial action the agent was doing.")
    
    thought_process: str = dspy.OutputField(desc="The reasoning behind the decision.")
    option: int = dspy.OutputField(desc="The chosen option (1 for Wait, 2 for Continue).")

def decide_to_react(context: str, current_time: str, agent: str, agent_with: str, agent_with_action: str, agent_observation: str, agent_with_observation: str, initial_action_description: str) -> int:
    try:
        predict = dspy.ChainOfThought(DecideToReactSignature)
        response = predict(
            context=context,
            current_time=current_time,
            agent=agent,
            agent_with=agent_with,
            agent_with_action=agent_with_action,
            agent_observation=agent_observation,
            agent_with_observation=agent_with_observation,
            initial_action_description=initial_action_description
        )
        # Ensure result is valid option
        opt = response.option
        if opt not in [1, 2]:
            return 2 # Default to continue
        return opt
    except Exception as e:
        print(f"Error in decide_to_react: {e}")
        return 2 # Default to continue

if __name__ == "__main__":
    print(decide_to_react(context="You are walking through the park going to Johns Pub.",
                          current_time="10:00 PM",
                          agent="William Strange",
                          agent_with="John Smith",
                          agent_with_action="Jumping up and down",
                          agent_observation="walking through the park",
                          agent_with_observation="Jumping up and down in the park",
                          initial_action_description="walking through the park going to Johns Pub"))
    