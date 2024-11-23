from pydantic import create_model

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="reflection_points")

template = """You are reflecting on the subjects in the statements. You need to determine the most salient high-level questions we can answer about the subjects in the statements.
{memory}

Given only the information above, what are {count} most salient high-level questions we can answer about the subjects in the statements?"""


def reflection_points(memory: str, count: int) -> list[str]:
    questions = {f"Option {i}": (str, ...) for i in range(1, count + 1)}

    ReflectionPoints = create_model("ReflectionPoints", **questions)

    structured_llm = llm.with_structured_output(ReflectionPoints)
    content = template.format(memory=memory, count=count)

    reflection_points = structured_llm.invoke([HumanMessage(content=content)])
    return [question for question in reflection_points.values()]

if __name__ == "__main__":
    print(reflection_points(memory="Volunteers, including Susan and Tom, spend weekends working on the garden;\nThey plant new vegetables and flowers, and install a small pond;\nThe garden becomes a beautiful spot for relaxation and education in the neighborhood.", count=2))
    print(reflection_points(memory="Given only the information above, what are 2 most salient high-level questions we can answer about the subjects in the statements?", count=2))
    print(reflection_points(memory="Alice Johnson is preparing for her final exams in university;\nMark has been helping Alice study by quizzing her on various subjects;\nAlice appreciates Mark's support and feels more confident about her exams;\nMark and Alice plan to celebrate together after the exams are over.", count=2))
    print(reflection_points(memory="Emma Thompson has started a new diet to improve her health;\nShe has been sharing her progress with her friend Sarah, who is a nutritionist;\nSarah offers advice and encourages Emma to stick with her goals;\nEmma feels motivated and grateful for Sarah's help and expertise.", count=4))
    print(reflection_points(memory="Robert Green is training for a marathon next month;\nHe runs early in the morning before work and tracks his progress;\nHis colleague, Linda, notices his dedication and asks for running tips;\nRobert is happy to share his training routine and motivate Linda to start running.", count=3))