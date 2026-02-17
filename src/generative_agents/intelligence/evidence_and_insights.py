import dspy


class EvidenceAndInsightsSignature(dspy.Signature):
    """
    Infer high-level standalone insights from statements.
    """

    statements: str = dspy.InputField(desc="List of statements.")
    number_of_insights: int = dspy.InputField(desc="Number of insights to generate.")

    insights: list[str] = dspy.OutputField(
        desc="List of insights inferred from the statements."
    )


def evidence_and_insights(statements: list[str], number_of_insights: int) -> list[str]:
    statements_str = "\n".join(
        [f"{i + 1}. {s.strip()}" for i, s in enumerate(statements)]
    )

    try:
        predict = dspy.ChainOfThought(EvidenceAndInsightsSignature)
        response = predict(
            statements=statements_str, number_of_insights=number_of_insights
        )
        # Ensure we get exactly the number requested if possible, or just return what we got
        return response.insights[:number_of_insights]
    except Exception as e:
        print(f"Error in evidence_and_insights: {e}")
        return [f"Insight {i + 1}" for i in range(number_of_insights)]


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(
            lm=dspy.DummyLM([{"insights": ["John is academic.", "John is social."]}])
        )
    print(
        evidence_and_insights(
            statements=[
                "John Rossi is a student",
                "John Rossi is learning a lot",
                "John Rossi likes discussing with his peers",
            ],
            number_of_insights=2,
        )
    )
