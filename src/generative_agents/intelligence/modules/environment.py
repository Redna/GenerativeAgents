import dspy
from generative_agents.common.dspy_config import fast_lm

class ActionConsequenceSignature(dspy.Signature):
    """
    Evaluate an agent's interaction with a world object. 
    Determine if the action succeeds, fails, or causes an environmental reaction.
    Keep consequences realistic and grounded. If the action is normal, output 'None'.
    """
    agent_name: str = dspy.InputField(desc="Name of the acting agent.")
    action_description: str = dspy.InputField(desc="What the agent is doing.")
    object_name: str = dspy.InputField(desc="The object being interacted with.")
    consequence: str = dspy.OutputField(
        desc="A brief, 3rd-person description of the environmental reaction, or 'None' if nothing special happens."
    )

class WorldPhysicsSimulator(dspy.Module):
    """Evaluates agent interactions to generate dynamic environmental feedback."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ActionConsequenceSignature)

    def forward(self, agent_name: str, action_description: str, object_name: str) -> str:
        try:
            with dspy.context(lm=fast_lm) if fast_lm else dspy.context():
                result = self.predict(
                    agent_name=agent_name, 
                    action_description=action_description, 
                    object_name=object_name
                )
            
            consequence = result.consequence.strip()
            if consequence.lower() in ["none", "null", "nothing", ""]:
                return ""
            return consequence
        except Exception:
            return ""
