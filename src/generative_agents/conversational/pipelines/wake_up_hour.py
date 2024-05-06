from pydantic import BaseModel, Field

from generative_agents.conversational.llm import llm

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline


template = """You are in a roleplay game and act as an agent. Your task is to estimate the wake up hour of an agent.
You are {{agent_name}}. Your identity is: 
{{agent_identity}}. 

{{agent_lifestyle}}.
    
When does {{agent_name}} wake up today?"""


class WakeUpHour(BaseModel):
    rationale: str = Field(description="maximum three sentences reason for the wake up hour")
    wake_up_hour: int = Field(gt=0, lt=13, description="time in 12-hour clock format")

def estimate_wake_up_hour(agent_name: str, agent_identity: str, agent_lifestyle: str) -> str:
    wake_up_hour = grammar_pipeline.run(model=WakeUpHour, prompt_template=template, template_variables={
        "agent_name": agent_name,
        "agent_identity": agent_identity,
        "agent_lifestyle": agent_lifestyle
    })

    return  str(wake_up_hour.wake_up_hour).zfill(2) + ":00 " + ("AM" if wake_up_hour.wake_up_hour < 12 else "PM")

    
if __name__ == "__main__":
    print(estimate_wake_up_hour("Emily Clark",
                                "Emily Clark is a 25 year old freelance graphic designer. She adores creating vibrant and unique designs. Emily is an avid traveler, seeking inspiration from different cultures around the world. She prefers working late nights to capture the essence of her travels in her designs.",
                   "Emily finds her creativity peaks in the quiet of the night."))

    print(estimate_wake_up_hour("Michael Thompson",
                                "Michael Thompson is a 40 year old professional photographer. He specializes in wildlife photography and spends months in remote locations capturing the beauty of nature. Michael is an adventure seeker and enjoys hiking and camping.",
                   "Michael is accustomed to waking up at dawn to catch the perfect light for his photographs."))
    
    print(estimate_wake_up_hour("Linda Wu",
                                "Linda Wu is a 35 year old entrepreneur running her own start-up. She is dedicated to creating eco-friendly products. Linda is passionate about sustainability and environmental conservation. She practices yoga daily to stay focused and energized.",
                   "Linda is an early riser, finding the morning the best time to plan her day and meditate."))
