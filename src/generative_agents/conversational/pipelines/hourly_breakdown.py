
from pydantic import BaseModel, Field, create_model

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline
from typing import Dict, List

from pydantic import BaseModel

hours = ["12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
         "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

template = """You are a character in a role play game. You are thinking about your day and create an hourly schedule. 
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.

You will act as {{name}}.
Your identity is: 
{{identity}}

Here is today's plan in broad-strokes:
{%- for start_time, activity in daily_plan.items() %}
{{loop.index}}.) {{activity}} at {{start_time}}
{%- endfor %}

How does {{name}}'s complete hourly schedule look for today? You must follow the schedule format above. {{name}}'s day starts at {{wake_up_hour}}. Before that, {{name}} is sleeping."""


def create_hourly_schedule(name: str, identity: str, daily_plan: list[dict[str, str]], wake_up_hour: str) -> str:
    # TODO use create_model like in task_decomposition.py - freeze the ones until wake_up_hour to make it fixed
    HourlySchedule = create_model("HourlySchedule", **{hour: (str, Field(..., description=f"Brief activity at this time. Must not be empty", min_length=2)) for hour in hours})

    schedule = grammar_pipeline.run(model=HourlySchedule, prompt_template=template, template_variables={
        "name": name,
        "identity": identity,
        "daily_plan": daily_plan, 
        "wake_up_hour": wake_up_hour
    })

    def to_schedule(schedule, wake_up_hour: int) -> List[Dict[str, str]]:
        wake_up_hour_index = hours.index(wake_up_hour.zfill(8))

        list_schedule = list()

        for i, (hour, activity) in enumerate(schedule.items()):
            if i < wake_up_hour_index:
                list_schedule.append({"time": hour, "activity": "Sleeping"})
            elif i == wake_up_hour_index:
                list_schedule.append({"time": hour, "activity": "Wake up and get ready for the day"})
            else: 
                list_schedule.append({"time": hour, "activity": activity})

        return list_schedule

    schedule = to_schedule(schedule.model_dump(), wake_up_hour)

    return schedule


if __name__ == "__main__":
        hourly_schedule = create_hourly_schedule(name="Emily Johnson", 
                                                 identity="Emily Johnson is a 28 year old graphic designer. She is creative and enjoys exploring new art forms. She lives with her two cats and enjoys gardening. Emily is a yoga enthusiast and likes to cook healthy meals. She is a morning person and enjoys waking up early. She is a social person and enjoys meeting new people.",
                                                 daily_plan=[
                                                     {"time": "08:00 AM", "activity": "breakfast"},
                                                     {"time": "09:00 AM", "activity": "work"},
                                                     {"time": "12:00 PM", "activity": "lunch"},
                                                     {"time": "01:00 PM", "activity": "work"},
                                                     {"time": "05:00 PM", "activity": "finish work and head to the town square for socializing and meeting new people"},
                                                     {"time": "07:00 PM", "activity": "visit a local pub for an evening beer"},
                                                     {"time": "09:00 PM", "activity": "head back home"},
                                                     {"time": "10:00 PM", "activity": "go to sleep"}
                                                 ])
        print(hourly_schedule)
        
        hourly_schedule = create_hourly_schedule(name="John Doe",
                                                    identity="John Doe is a 35 year old software developer. He is passionate about technology and loves coding. He lives in a quiet suburb and enjoys the peace it offers. John is an avid reader and spends his evenings reading tech articles. He prefers a structured day and enjoys the solitude of working from home.",
                                                    daily_plan=[
                                                        {"time": "07:00 AM", "activity": "morning run"},
                                                        {"time": "08:00 AM", "activity": "breakfast and reading tech news"},
                                                        {"time": "09:00 AM", "activity": "start work"},
                                                        {"time": "12:00 PM", "activity": "lunch break and a short walk"},
                                                        {"time": "01:00 PM", "activity": "continue work"},
                                                        {"time": "06:00 PM", "activity": "end work and relax with a book"},
                                                        {"time": "08:00 PM", "activity": "dinner"},
                                                        {"time": "09:00 PM", "activity": "leisure time or hobby projects"},
                                                        {"time": "11:00 PM", "activity": "go to sleep"}
                                                    ])
        print(hourly_schedule)

        hourly_schedule = create_hourly_schedule(name="Sarah Lee",
                                                    identity="Sarah Lee is a 30 year old artist. She is deeply passionate about painting and spends most of her day in her studio. She loves nature and often takes long walks to find inspiration. Sarah is a night owl and finds herself most creative during the late hours.",
                                                    daily_plan=[
                                                        {"time": "10:00 AM", "activity": "breakfast and morning meditation"},
                                                        {"time": "11:00 AM", "activity": "studio time for painting"},
                                                        {"time": "02:00 PM", "activity": "lunch and a walk in the park"},
                                                        {"time": "03:00 PM", "activity": "return to studio work"},
                                                        {"time": "07:00 PM", "activity": "dinner and socialize with friends"},
                                                        {"time": "09:00 PM", "activity": "evening studio session"},
                                                        {"time": "01:00 AM", "activity": "relaxation and bedtime routine"},
                                                        {"time": "02:00 AM", "activity": "go to sleep"}
                                                    ])
        print(hourly_schedule)
