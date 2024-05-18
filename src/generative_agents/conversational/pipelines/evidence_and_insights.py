from enum import Enum
from pydantic import BaseModel, Field, create_model

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You are infering standalone insights from statments.

Input:
{%- for statement in statements %}
    {{ loop.index }}. {{statement | trim}}
{%- endfor %}
What {{number_of_insights}} high-level standalone insights can you infer from the above statements?"""

def evidence_and_insights(statements: list[str], number_of_insights: int) -> list[str]:
    insights = {f"Insight {i}": (str, ...) for i in range(1, number_of_insights + 1)}

    NumberOfInsights = create_model("ReflectionPoints", **insights)

    reflection_points = grammar_pipeline.run(model=NumberOfInsights, prompt_template=template, template_variables={
        "statements": statements,
        "number_of_insights": number_of_insights
    })

    return [insight for insight in reflection_points.values()]

if __name__ == "__main__":
    print(evidence_and_insights(statements=["John Rossi is a student", "John Rossi is learning a lot",
                    "John Rossi likes discussing with his peers"], number_of_insights=2))
    print(evidence_and_insights(statements=["Sara Johnson is an engineer", "Sara Johnson works on innovative projects",
                                "Sara Johnson enjoys collaborating with her team"], number_of_insights=2))
    print(evidence_and_insights(statements=["David Smith is a chef", "David Smith specializes in Italian cuisine",
                                "David Smith values using fresh ingredients"], number_of_insights=2))
    print(evidence_and_insights(statements=[
                                "Alex Martinez is a biologist", "Alex Martinez studies marine life"], number_of_insights=1))
    
