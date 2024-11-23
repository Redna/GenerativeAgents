
from copy import deepcopy
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from generative_agents.utils import time_string_to_time

llm = ChatGroq(model="llama3-8b-8192", max_tokens=4069, name="hourly_breakdown")

hours = ["12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
         "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

template = """You act as {name}. {name} identity is:
{identity}

You are thinking about your day and create an hourly schedule.
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.

Here is today's plan in broad-strokes:
{formatted_daily_plan}

{name}'s day starts at {wake_up_hour}. Before that, {name} is sleeping.
This is how {name}'s hourly schedule looks like. Start with {wake_up_hour} and fill in the rest of the day:
{schedule}"""


def create_hourly_schedule(name: str, identity: str, daily_plan: list[dict[str, str]], wake_up_hour: str) -> str:
    wake_up_hour_index = hours.index(wake_up_hour.zfill(8))

    current_schedule = [(hour, "sleeping") for hour in hours if time_string_to_time(hour).hour < time_string_to_time(wake_up_hour).hour]

    current_schedule += [(hours[wake_up_hour_index], "wake up and get ready for the day")]

    schedule_template = deepcopy(current_schedule)
    schedule_template += [[hour, "<fill in>"] for hour in hours[wake_up_hour_index+1:]]

    formatted_schedule = '\n'.join([f"{hour} - {action}" for hour, action in schedule_template])
    formatted_schedule_current_schedule = '\n'.join([f"{hour} - {action}" for hour, action in current_schedule[:-1]])

    content = template.format(name=name, identity=identity, formatted_daily_plan=daily_plan, wake_up_hour=wake_up_hour, schedule=formatted_schedule)

    preamble = ("Here is my hourly schedule for today\n\n"
                f"{formatted_schedule_current_schedule}\n")

    generated_schedule = llm.invoke([HumanMessage(content=content), AIMessage(content=preamble)])

    generated_schedule_list = [tuple(entry.split(" - ")) for entry in generated_schedule.content.split("\n")]

    first_hour = generated_schedule_list[0][0]

    if wake_up_hour == first_hour:
        current_schedule[-1] = (first_hour, generated_schedule_list.pop(0)[1])

    for hour in hours[wake_up_hour_index+1:]:
        if not generated_schedule_list:
            current_schedule.append((hour, "sleeping"))
            continue

        if hour == generated_schedule_list[0][0]:
            current_schedule.append((hour, generated_schedule_list.pop(0)[1]))
        else:
            current_schedule.append((hour, current_schedule[-1][1]))


    return [{'hour': hour, 'activity': activity} for hour, activity in current_schedule]


if __name__ == "__main__":
        hourly_schedule = create_hourly_schedule(name="Emily Johnson",
                                                 wake_up_hour="06:00 AM",
                                                 identity="Emily Johnson is a 28 year old graphic designer. She is creative and enjoys exploring new art forms. She lives with her two cats and enjoys gardening. Emily is a yoga enthusiast and likes to cook healthy meals. She is a morning person and enjoys waking up early. She is a social person and enjoys meeting new people.",
                                                 daily_plan={"06:00 AM": "wake up and morning yoga",
                                                             "07:00 AM": "breakfast and coffee",
                                                             "08:00 AM": "work",
                                                             "12:00 PM": "lunch",
                                                             "01:00 PM": "work",
                                                             "05:00 PM": "finish work and head to the town square for socializing and meeting new people",
                                                             "07:00 PM": "visit a local pub for an evening beer",
                                                             "09:00 PM": "head back home",
                                                             "10:00 PM": "go to sleep"})
        print(hourly_schedule)

        hourly_schedule = create_hourly_schedule(name="John Doe",
                                                    wake_up_hour="06:00 AM",
                                                    identity="John Doe is a 35 year old software developer. He is passionate about technology and loves coding. He lives in a quiet suburb and enjoys the peace it offers. John is an avid reader and spends his evenings reading tech articles. He prefers a structured day and enjoys the solitude of working from home.",
                                                    daily_plan={"06:00 AM": "wake up and morning routine",
                                                                "06:30 AM": "breakfast and coffee",
                                                                "07:00 AM": "start work",
                                                                "12:00 PM": "lunch break",
                                                                "01:00 PM": "continue work",
                                                                "06:00 PM": "end work and relax",
                                                                "08:00 PM": "dinner",
                                                                "09:00 PM": "read tech articles",
                                                                "11:00 PM": "go to sleep"}
        )
        print(hourly_schedule)