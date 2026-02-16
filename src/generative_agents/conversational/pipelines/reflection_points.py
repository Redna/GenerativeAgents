import dspy

class ReflectionPointsSignature(dspy.Signature):
    """
    Determine the most salient high-level questions we can answer about the subjects in the statements.
    """
    memory: str = dspy.InputField(desc="Statements about subjects.")
    count: int = dspy.InputField(desc="Number of questions to generate.")
    
    questions: list[str] = dspy.OutputField(desc="List of salient questions.")

def reflection_points(memory: str, count: int) -> list[str]:
    try:
        predict = dspy.ChainOfThought(ReflectionPointsSignature)
        response = predict(
            memory=memory,
            count=count
        )
        return response.questions[:count]
    except Exception as e:
        print(f"Error in reflection_points: {e}")
        return [f"Question {i+1}?" for i in range(count)]

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "questions": ["What is x?", "Who is y?"]
         }]))
    print(reflection_points(memory="Volunteers...", count=2))