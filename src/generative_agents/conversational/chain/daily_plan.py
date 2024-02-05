import asyncio
from datetime import datetime
import re
from langchain.chains import SequentialChain
from pydantic import BaseModel
import yaml
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are a character in a role-play game. Given a name and an identity, you are supposed to act like you are that character."""

remember_user = """You are {name}. Your identity is:
{identity}

[Statements]
{statements}

Given the statements above, what is the most important thing that {name} should remember as they plan for {today}?
If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)
Write the response from {name}'s perspective and be as brief as possible."""


class _Remember(BaseModel):
    statements: str
    name: str
    today: str
    identity: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(remember_user)])
        _things_i_should_remember_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 200,
            "top_p": 0.96,
            "temperature": 0.4}, verbose=True)

        completion = await _things_i_should_remember_chain.ainvoke(input={"statements": self.statements,
                                                                          "name": self.name,
                                                                          "today": self.today,
                                                                          "identity": self.identity})
        return completion["text"]


feel_user = """You are {name}. Your identity is:
{identity}

[Statements]
{statements}

Given the statements above, how might we summarize {name}'s feelings about their days up to now?
Write the response from {name}'s perspective and be as brief as possible."""


class _Feelings(BaseModel):
    statements: str
    name: str
    identity: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(feel_user)])
        _things_i_should_remember_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 200,
            "top_p": 0.96,
            "temperature": 0.4}, verbose=True)

        completion = await _things_i_should_remember_chain.ainvoke(input={"statements": self.statements,
                                                                          "name": self.name,
                                                                          "identity": self.identity})
        return completion["text"]


status_user = """You are {name}. Your identity is:
{identity}

{name}'s status from {yesterday}:
{current_activity}

{name}'s thoughts at the end of {yesterday}:
{thought_note} {plan_note}
It is now {today}. Given the above, what is {name}'s status for {today} that reflects {name}'s thoughts at the end of {yesterday}? Write this in third-person talking about {name}.
If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement). Be as brief as possible."""


class _Currently(BaseModel):
    name: str
    yesterday: str
    today: str
    current_activity: str
    thought_note: str
    plan_note: str
    identity: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(status_user)])
        _currently_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 650,
            "top_p": 0.96,
            "temperature": 0.4}, verbose=True)

        completion = await _currently_chain.ainvoke(input={"name": self.name,
                                                           "yesterday": self.yesterday,
                                                           "today": self.today,
                                                           "current_activity": self.current_activity,
                                                           "thought_note": self.thought_note,
                                                           "plan_note": self.plan_note,
                                                           "identity": self.identity})

        return completion["text"]


plan_system = system + """
Output format:
    ```yaml
    my_plan_for_today: 
        - <activity 1>: <time in 12-hour clock format>
        - <activity 2>: <time in 12-hour clock format>
        - <activity 3>: <time in 12-hour clock format>
        # ...
        - <activity n>: <time in 12-hour clockformat>
    ```"""
plan_user_shot_1 = """You are John Doe. Your identity is:
John Doe is a 30 year old software developer. He enjoys coding and is a coffee enthusiast. John loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.

John Doe has reflected and planned the following for today based on your feelings yesterday: 
On 2023-08-01, John Doe's status reflects his excitement to follow up on the intriguing conversation about AI trends with fellow software developer Julia Brown. He is looking forward to their scheduled meeting at their regular coffee shop on August 1, 2023, as he believes it will provide valuable insights and opportunities for collaboration in the field of artificial intelligence.

Today is 2023-08-01. What is John Doe's plan today in broad-strokes? (Mention for each activity the time in 12-hour clock format.)"""

plan_ai_shot_1 = """```yaml
my_plan_for_today:
    - Wake up and have breakfast: 6:30 AM
    - Start work on a new project: 7:30 AM
    - Take a short break for coffee: 10:30 AM
    - Meet with fellow software developer Julia Brown to discuss AI trends: 12:00 PM
    - Resume work on the current project: 3:00 PM
    - Read a sci-fi novel for leisure: 7:00 PM
    - Play chess with friends or online: 9:00 PM
```"""

plan_user = """You are {name}. Your identity is:
{identity}

{name} has reflected and planned the following for today based on your feelings yesterday: 
{feelings_for_today}

Today is {today}. What is {name}'s plan today in broad-strokes? (Mention for each activity the time in 12-hour clock format.)"""


class _Plan(BaseModel):
    identity: str
    today: str
    name: str
    feelings_for_today: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(plan_system),
            HumanMessagePromptTemplate.from_template(plan_user_shot_1),
            AIMessagePromptTemplate.from_template(plan_ai_shot_1),
            HumanMessagePromptTemplate.from_template(plan_user)])
        _daily_plan_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 650,
            "top_p": 0.96,
            "temperature": 0.4}, verbose=True)

        completion = await _daily_plan_chain.ainvoke(input={"identity": self.identity,
                                                            "today": self.today,
                                                            "name": self.name,
                                                            "feelings_for_today": self.feelings_for_today})

        pattern = r'```yaml(.*)```'
        match = re.search(pattern, completion["text"], re.DOTALL)
        if match:
            try:
                output = yaml.safe_load(match.group(1))

                def get_time(item):
                    _, time_str = item
                    return datetime.strptime(time_str, "%I:%M %p")

                daily_plan = output["my_plan_for_today"]

                sanitized_daily_plan = []
                for entry in daily_plan:
                    for activity, time in entry.items():
                        match = re.search(r"\d{1,2}:\d{2} [AP]M", time)
                        if match:
                            time = match.group()
                        else:
                            continue
                        sanitized_daily_plan.append((activity, time))

                sanitized_daily_plan = sorted(
                    sanitized_daily_plan, key=get_time)
                return sanitized_daily_plan
            except Exception as error:
                pass
        return []


class DailyPlanAndStatus(BaseModel):
    name: str
    identity: str
    today: str
    yesterday: str
    statements: str
    current_activity: str

    async def run(self):
        remember, feelings = await asyncio.gather(
            _Remember(statements=self.statements, name=self.name,
                      today=self.today, identity=self.identity).run(),
            _Feelings(statements=self.statements, name=self.name,
                      identity=self.identity).run(),
        )

        feelings_for_today = await _Currently(name=self.name, yesterday=self.yesterday, today=self.today, current_activity=self.current_activity, thought_note=feelings, plan_note=remember, identity=self.identity).run()

        daily_plan = await _Plan(identity=self.identity, feelings_for_today=feelings_for_today, today=self.today, name=self.name).run()

        plan = " ".join([f"{i+1}) {activity} at {time}" for i,
                        (activity, time) in enumerate(daily_plan)])

        return plan, feelings_for_today


async def __tests():
    t = [
        DailyPlanAndStatus(
            name="John Doe",
            identity="John Doe is a 30 year old software developer. He enjoys coding and is a coffee enthusiast. John loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.",
            today="2023-08-01",
            yesterday="2023-07-31",
            statements="""Yesterday was productive; I managed to solve a complex coding problem at work and felt a real sense of achievement.
    I met Julia Brown for coffee; we had a fascinating conversation about the latest trends in artificial intelligence.
    In the evening, I read a chapter of 'The Martian' by Andy Weir. The ingenuity of the protagonist always inspires me.
    I tried a new chess strategy I learned from a YouTube tutorial, and it worked surprisingly well in my online game.
    I experimented with brewing Ethiopian coffee; its unique flavor made the morning even more enjoyable.""",
            current_activity="John Doe loves the morning as he feels super relaxed."
        ).run(),
        DailyPlanAndStatus(
            name="John Doe",
            identity="John Doe is a 30 year old software developer with a passion for technology and innovation. He's also a fan of outdoor activities.",
            today="2023-08-02",
            yesterday="2023-08-01",
            statements="""The hiking trip yesterday was breathtaking; the view from the peak was unforgettable.
    I finally finished reading 'Neuromancer' by William Gibson; the cyberpunk world has sparked some new project ideas.
    During lunch, I brainstormed with a colleague about a potential app that could solve a common problem in our field.
    I started learning Spanish on a language app; understanding the basics was more enjoyable than expected.
    The homemade pizza I made for dinner was a hit; experimenting with different toppings was fun.""",
            current_activity="Feeling energized for today's coding challenges."
        ).run(),
        DailyPlanAndStatus(
            name="John Doe",
            identity="John Doe, a 30 year old software developer, is also an avid traveler and food lover.",
            today="2023-08-03",
            yesterday="2023-08-02",
            statements="""Yesterday's team meeting was highly productive; we outlined the roadmap for our upcoming project.
    I had a delightful dinner at a new Thai restaurant; the Pad Thai was authentic and reminded me of my trip to Bangkok.
    I watched a documentary on renewable energy solutions; it's fascinating to learn about sustainable living.
    I started sketching again after a long break; it's refreshing to engage in creative activities outside of work.
    I helped a friend troubleshoot some issues with her website; it's rewarding to use my skills to assist others.""",
            current_activity="Looking forward to a day full of learning and creativity."
        ).run(),
        DailyPlanAndStatus(
            name="John Doe",
            identity="Besides being a software developer, John Doe is deeply interested in music and plays the guitar.",
            today="2023-08-04",
            yesterday="2023-08-03",
            statements="""I had an impromptu jam session with friends yesterday; playing music together is always a joy.
    I discovered a new coding library that could significantly improve my current project's performance.
    I attended a webinar on blockchain technology; the potential applications beyond cryptocurrencies are intriguing.
    I experimented with a new recipe for banana bread; adding dark chocolate chips made it exceptionally delicious.
    I planned my next travel destination; exploring Japan has been on my bucket list for years.""",
            current_activity="Excited to dive into today's coding tasks and maybe sneak in some guitar practice."
        ).run(),
        DailyPlanAndStatus(
            name="John Doe",
            identity="John Doe is not just a software developer; he's also a fitness enthusiast and enjoys working out.",
            today="2023-08-05",
            yesterday="2023-08-04",
            statements="""Yesterday's workout was intense; I hit a new personal record on the deadlift.
    I started reading 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari; it's thought-provoking.
    I explored a new VR game; the immersive experience is unlike anything I've played before.
    I had a video call with family; catching up with them always brightens my day.
    I signed up for a local coding hackathon; I'm excited to meet other developers and exchange ideas.""",
            current_activity="Eager to tackle today's challenges and maybe fit in a quick run."
        ).run()
    ]
    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
