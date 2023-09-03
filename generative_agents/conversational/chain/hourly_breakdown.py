from typing import List, Tuple

from pydantic import BaseModel
from llm import llm
from langchain import LLMChain, PromptTemplate

hours = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
         "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
         "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
         "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
         "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

_template = """
Hourly schedule format:
{hourly_schedule_format}

{identity}

Here the originally intended hourly breakdown 
{name}'s schedule for today:
{intended_schedule}

{prior_schedule}
{current_activity}
"""

_prompt = PromptTemplate(input_variables=["hourly_schedule_format", "identity", "name", "intended_schedule", "prior_schedule", "current_activity"],
                         template=_template)

class HourlyBreakdown(BaseModel):
    
    identity: str
    current_hour: str
    name: str
    today: str
    hourly_organized_activities: List[str]
    actual_activities: List[str]
    current_activity: str

    def _build_hourly_schedule_format(self) -> str:
        formatted_hours += [f"{self.today} -- {hour} | Activity: [Fill in]" for hour in hours]
        return "\n".join(formatted_hours)

    def _build_intended_schedule(self) -> str: 
        formatted_hours = [f"{self.today} -- {hour} | Activity: {activity}" for hour, activity in zip(hours, self.hourly_organized_activities)]
        return "\n".join(formatted_hours)

    def _build_prior_schedule(self) -> str: 
        formatted_hours = [f"{self.today} -- {hour} | Activity: {activity}" for hour, activity in zip(hours, self.actual_activities)]
        return "\n".join(formatted_hours)

    def _build_current_activity(self) -> str: 
        return f"{self.current_hour} | Activity: {self.name} is {self.current_activity}"

    def run(self): 
        chain = LLMChain(prompt=_prompt, llm=llm)
        
        hourly_schedule_format = self._build_hourly_schedule_format()
        intended_schedule = self._build_intended_schedule()
        prior_schedule = self._build_prior_schedule()
        current_activity = self._build_current_activity()
        
        return chain.run(hourly_schedule_format=hourly_schedule_format, 
                         identity=self.identity, 
                         name=self.name, 
                         intended_schedule=intended_schedule, 
                         prior_schedule=prior_schedule, 
                         current_activity=current_activity)


