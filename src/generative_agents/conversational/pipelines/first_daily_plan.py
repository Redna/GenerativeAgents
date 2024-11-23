from pydantic import BaseModel, Field

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="daily_plan")

template = """You are {name}. Your identity is:
{identity}

Today is {today}. What is {name}'s plan today in broad-strokes?
{name} will wake up and complete the morning routine at {wake_up_hour}"""


class PlanOutline(BaseModel):
    hour: int = Field(description="The hour of the day when the activity needs to happen. MUST be between 1 and 24. (1-24)", ge=1, le=24)
    description: str = Field(
        description="A brief description of the activity planned for the hour.")


class DailyPlan(BaseModel):
    plan_in_broad_strokes: list[PlanOutline] = Field(
        description="The plan in broad-stokes for the day with activities and their scheduled times.")


def create_daily_plan(name: str, identity: str, today: str, wake_up_hour: str) -> list[dict[str, str]]:
    structured_llm = llm.with_structured_output(DailyPlan)

    content = template.format(name=name, identity=identity, today=today, wake_up_hour=wake_up_hour)
    plan = structured_llm.with_retry(stop_after_attempt=3).invoke([HumanMessage(content=content)])

    converted_plan = {}

    for activity in plan.plan_in_broad_strokes:
        activity.hour = activity.hour % 12

        meridien = "AM" if activity.hour < 12 else "PM"

        hour = activity.hour
        if activity.hour == 0:
            hour = 12

        hour = f"{str(hour).zfill(2)}:00 {meridien}"

        converted_plan[hour] = activity.description

    return converted_plan
