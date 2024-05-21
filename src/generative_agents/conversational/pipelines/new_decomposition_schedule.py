
import datetime
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline
from generative_agents.utils import hour_string_to_time


template = """You are a character in a role play game. You are thinking about your day and create an hourly schedule. 
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.
Output a schedule in a valid json format.

You will act as {{name}}.
Your identity is: 
{{identity}}

Here is today's plan in broad-strokes:
{%- for daily_plan_item in daily_plan %}
{{loop.index}}.) {{daily_plan_item.activity}} at {{daily_plan_item.time}}
{%- endfor %}

How does {{name}}'s complete hourly schedule look for today? You must follow the schedule format above. {{name}}'s day starts at {{wake_up_hour}}. Before that, {{name}} is sleeping."""


class Task(BaseModel):
    time: str = Field(pattern="^(0[0-9]|1[0-2]):[0-5][0-9] (AM|PM)$", description="The time of the day in 12-hour clock format (hh:mm AM/PM)")
    activity: str = Field(description="The activity planned for the time.")

class NewDecompositionSchedule(BaseModel):
    schedule: list[Task] = Field(description="The list of tasks for the day with their scheduled times.")
    
def _generate_schedule(start_hour, schedule_slice) -> list[tuple[str, str, int]]:
    start_hour = hour_string_to_time(start_hour)
    for_time = start_hour

    schedule_with_hour = []
    for action, duration in schedule_slice:
        schedule_with_hour.append((for_time.strftime("%I:%M %p"), action, duration))
        for_time += datetime.timedelta(minutes=int(duration))

    return schedule_with_hour

def create_new_decomposition_schedule(agent: str, start_hour: int, end_hour: int, new_event: str, new_event_duration: int, schedule_slice: list[tuple[str, int]]) -> list[tuple[str, int]]:
    schedule = _generate_schedule(start_hour, schedule_slice)
    start_hour_str = hour_string_to_time(start_hour).strftime("%I:%M %p")
    end_hour_str = hour_string_to_time(end_hour).strftime("%I:%M %p")

    schedule = grammar_pipeline.run(model=NewDecompositionSchedule, prompt_template=template, template_variables={
        "agent": agent,
        "start_hour": start_hour_str,
        "end_hour": end_hour_str,
        "new_event": new_event,
        "new_event_duration": new_event_duration,
        "schedule_slice": schedule_slice
    })

    return schedule.schedule

if __name__ == "__main__":
        print(create_new_decomposition_schedule(agent="Jane Doe", 
                         start_hour=9, end_hour=13, 
                         new_event="Team meeting about quarterly goals.", 
                         new_event_duration=30,
                         schedule_slice=[("email correspondence", 45), 
                                         ("project work", 60), 
                                         ("coffee break", 15),
                                         ("project work", 60),
                                         ("quick team catch-up", 20),
                                         ("administrative tasks", 40)
                         ]))
        print(create_new_decomposition_schedule(agent="Alex Smith",
                            start_hour=8, end_hour=11, 
                            new_event="Training session on new software tools.", 
                            new_event_duration=45,
                            schedule_slice=[("checking emails", 30), 
                                            ("team briefing", 15), 
                                            ("work on current project", 60),
                                            ("coffee break", 15),
                                            ("preparation for training", 60)
                            ]))
        print(create_new_decomposition_schedule(agent="Michelle Lee",
                            start_hour=13, end_hour=17, 
                            new_event="Client meeting to discuss project updates.", 
                            new_event_duration=25,
                            schedule_slice=[("project development", 60), 
                                            ("lunch break", 45), 
                                            ("internal review meeting", 30),
                                            ("preparing meeting materials", 60),
                                            ("end-of-day wrap-up", 45)
                            ]))
        print(create_new_decomposition_schedule(agent="Carlos Rodriguez",
                            start_hour=12, end_hour=16, 
                            new_event="Workshop on effective communication skills.", 
                            new_event_duration=120,
                            schedule_slice=[("team stand-up meeting", 15), 
                                            ("project work", 60), 
                                            ("lunch break", 45),
                                            ("feedback session", 60),
                                            ("preparing for workshop", 60)
                            ]))
        