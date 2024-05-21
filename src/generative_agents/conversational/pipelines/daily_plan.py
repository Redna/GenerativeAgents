
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline


template = """You act as {{name}} in a roleplay game. Your identity is:
{{identity}}

[Statements]
{{statements}}

Given the statements above, what is the most important thing that {{name}} should remember as they plan for {{today}}?
If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)
Write the response from {{name}}'s perspective and be as brief as possible."""


class Remember(BaseModel):
    things_i_should_remember: str = Field(
        description="Contains a brief overview of things I need to remember to create my daily plan.")


def find_things_to_remember(name: str, identity: str, statements: str, today: str) -> str:
    remember = grammar_pipeline.run(model=Remember, prompt_template=template, template_variables={
        "name": name,
        "identity": identity,
        "statements": statements,
        "today": today
    })

    return remember.things_i_should_remember


template = """You are {{name}}. Your identity is:
{{identity}}

[Statements]
{{statements}}

Given the statements above, how might we summarize {{name}}'s feelings about their days up to now?
Write the response from {{name}}'s perspective and be as brief as possible."""


class Feelings(BaseModel):
    feelings: str = Field(
        description="Contains a brief summary of the agent's feelings about their days up to now.")


def find_feelings(name: str, identity: str, statements: str) -> str:
    feelings = grammar_pipeline.run(model=Feelings, prompt_template=template, template_variables={
        "name": name,
        "identity": identity,
        "statements": statements
    })

    return feelings.feelings


template = """You are {{name} acting in a role play game. Your identity is:
{{identity}}

{{name}}'s status from {{yesterday}}:
{{current_activity}}

{{name}}'s thoughts at the end of {{yesterday}}:
{{thought_note}} {{plan_note}}
It is now {{today}}. Given the above, what is {{name}}'s status for {{today}} that reflects {{name}}'s thoughts at the end of {{yesterday}}? Write this in third-person talking about {{name}}.
If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement). Be as brief as possible."""


class Reflections(BaseModel):
    reflection_of_yestereday: str = Field(
        description="Contains a brief reflection of the agent's thoughts at the end of yesterday written in the third-person.")


def define_current_status(name: str, yesterday: str, today: str, current_activity: str, thought_note: str, plan_note: str, identity: str) -> str:
    status = grammar_pipeline.run(model=Reflections, prompt_template=template, template_variables={
        "name": name,
        "yesterday": yesterday,
        "today": today,
        "current_activity": current_activity,
        "thought_note": thought_note,
        "plan_note": plan_note,
        "identity": identity
    })

    return status.status


template = """You are {{name}}. Your identity is:
{{identity}}

{{name}} has reflected and planned the following for today based on your feelings yesterday: 
{{feelings_for_today}}

Today is {{today}}. What is {{name}}'s plan today in broad-strokes? (Mention for each activity the time in 12-hour clock format.)"""


class PlanOutline(BaseModel):
    hour: int = Field(description="The hour of the day. (0-23)", ge=0, le=23)
    description: str = Field(
        description="A brief description of the activity planned for the hour.")


class DailyPlan(BaseModel):
    plan_in_broad_strokes: list[PlanOutline] = Field(
        description="The plan in broad-stokes for the day with activities and their scheduled times.")


def create_daily_plan(name: str, identity: str, today: str, feelings_for_today: str) -> list[dict[str, str]]:
    plan = grammar_pipeline.run(model=DailyPlan, prompt_template=template, template_variables={
        "name": name,
        "identity": identity,
        "today": today,
        "feelings_for_today": feelings_for_today
    })

    # convert 24-hour (int) clock to 12-hour clock

    converted_plan = {}

    for activity in plan.plan_in_broad_strokes:
        activity.hour = activity.hour % 12

        meridien = "AM" if activity.hour < 12 else "PM"

        hour = activity.hour
        if activity.hour == 0:
            hour = 12

        hour = f"{hour.zfill(2)}:00 {meridien}"

        converted_plan[hour] = activity.description

    return converted_plan


def create_daily_plan_and_status(name: str, identity: str, today: str, yesterday: str, statements: str, current_activity: str) -> tuple[str, str]:
    remember = find_things_to_remember(name, identity, statements, today)
    feelings = find_feelings(name, identity, statements)

    feelings_for_today = define_current_status(
        name, yesterday, today, current_activity, feelings, remember, identity)

    daily_plan = create_daily_plan(name, identity, today, feelings_for_today)

    plan = " ".join([f"{i+1}) {activity} at {time}" for i,
                    (time, activity) in enumerate(daily_plan.items())])

    return plan, feelings_for_today


if __name__ == "__main__":
    print(create_daily_plan_and_status(name="John Doe",
                                       identity="John Doe is a 30 year old software developer. He enjoys coding and is a coffee enthusiast. John loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.",
                                       today="2023-08-01",
                                       yesterday="2023-07-31",
                                       statements="""Yesterday was productive; I managed to solve a complex coding problem at work and felt a real sense of achievement."""))
    print(create_daily_plan_and_status(
        name="John Doe",
        identity="John Doe, a 30 year old software developer, is also an avid traveler and food lover.",
        today="2023-08-03",
        yesterday="2023-08-02",
        statements="""Yesterday's team meeting was highly productive; we outlined the roadmap for our upcoming project.
    I had a delightful dinner at a new Thai restaurant; the Pad Thai was authentic and reminded me of my trip to Bangkok.
    I watched a documentary on renewable energy solutions; it's fascinating to learn about sustainable living.
    I started sketching again after a long break; it's refreshing to engage in creative activities outside of work.
    I helped a friend troubleshoot some issues with her website; it's rewarding to use my skills to assist others.""",
        current_activity="Looking forward to a day full of learning and creativity."))

    print(create_daily_plan_and_status(name="John Doe",
                                       identity="Besides being a software developer, John Doe is deeply interested in music and plays the guitar.",
                                       today="2023-08-04",
                                       yesterday="2023-08-03",
                                       statements="""I had an impromptu jam session with friends yesterday; playing music together is always a joy.
    I discovered a new coding library that could significantly improve my current project's performance.
    I attended a webinar on blockchain technology; the potential applications beyond cryptocurrencies are intriguing.
    I experimented with a new recipe for banana bread; adding dark chocolate chips made it exceptionally delicious.
    I planned my next travel destination; exploring Japan has been on my bucket list for years.""",
                                       current_activity="Excited to dive into today's coding tasks and maybe sneak in some guitar practice."))
