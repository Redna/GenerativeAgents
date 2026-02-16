import dspy

class ChatRelationshipSignature(dspy.Signature):
    """
    Summarize what the agent feels or knows about their relationship with another agent based on statements.
    """
    statements: str = dspy.InputField(desc="Statements about interactions.")
    agent: str = dspy.InputField(desc="Name of the agent.")
    agent_with: str = dspy.InputField(desc="Name of the other agent.")
    
    relationship_summary: str = dspy.OutputField(desc="Summary of the relationship.")

def summarize_chat_relationship(statements: str, agent: str, agent_with: str) -> str:
    try:
        predict = dspy.ChainOfThought(ChatRelationshipSignature)
        response = predict(
            statements=statements,
            agent=agent,
            agent_with=agent_with
        )
        return response.relationship_summary
    except Exception as e:
        print(f"Error in summarize_chat_relationship: {e}")
        return "They know each other."

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "relationship_summary": "They are colleagues."
         }]))
    print(summarize_chat_relationship(statements="...", agent="Leo", agent_with="Mia"))