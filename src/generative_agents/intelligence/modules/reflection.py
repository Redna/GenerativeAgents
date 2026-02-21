import dspy
from typing import List

from generative_agents.common.dspy_config import thinking_lm

# --- Signatures ---

class ReflectionPointsSignature(dspy.Signature):
    """Determine the most salient high-level questions we can answer about the subjects in the statements."""
    memory: str = dspy.InputField(desc="Statements about subjects.")
    count: int = dspy.InputField(desc="Number of questions to generate.")
    questions: list[str] = dspy.OutputField(desc="List of salient questions.")

class EvidenceAndInsightsSignature(dspy.Signature):
    """Infer high-level standalone insights from statements."""
    statements: str = dspy.InputField(desc="List of statements.")
    number_of_insights: int = dspy.InputField(desc="Number of insights to generate.")
    insights: list[str] = dspy.OutputField(desc="List of insights inferred from the statements.")

class IdentitySignature(dspy.Signature):
    """Write a concise description about the agent's personality, family situation and characteristics."""
    agent: str = dspy.InputField(desc="Name of the agent.")
    context: str = dspy.InputField(desc="Context about the agent.")
    identity: str = dspy.OutputField(desc="Concise identity description.")

# --- Modules ---

class ReflectionPointGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ReflectionPointsSignature)

    def forward(self, memory: str, count: int) -> List[str]:
        try:
            with dspy.context(lm=thinking_lm) if thinking_lm else dspy.context():
                response = self.predict(memory=memory, count=count)
            return response.questions[:count]
        except Exception:
            return [f"Question {i + 1}?" for i in range(count)]

class InsightGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(EvidenceAndInsightsSignature)

    def forward(self, statements: List[str], number_of_insights: int) -> List[str]:
        statements_str = "\n".join(
            [f"{i + 1}. {s.strip()}" for i, s in enumerate(statements)]
        )
        try:
            with dspy.context(lm=thinking_lm) if thinking_lm else dspy.context():
                response = self.predict(
                    statements=statements_str, number_of_insights=number_of_insights
                )
            return response.insights[:number_of_insights]
        except Exception:
            return [f"Insight {i + 1}" for i in range(number_of_insights)]

class IdentityFormulator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(IdentitySignature)

    def forward(self, agent: str, context: str) -> str:
        try:
            with dspy.context(lm=thinking_lm) if thinking_lm else dspy.context():
                response = self.predict(agent=agent, context=context)
            return response.identity
        except Exception:
            return f"{agent} is a person."
