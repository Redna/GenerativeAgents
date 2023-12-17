import asyncio
import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

# TODO rewrite few shot prompt to match the new identity set
_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.
You will act as the agent {name}.
<|user|>

Output format: Output a valid json of the following format:
```
{{
    "total_duration": "<total duration in minutes>",
    "subtasks": [
        {{ 
            "duration": "<duration in minutes>",
            "remaining_minutes": "<remaining minutes after completing this subtask>",
            "activity": "<activity in one brief sentence>"
        }},
        {{ 
            "duration": "<duration in minutes>",
            "remaining_minutes": "<remaining minutes after completing this subtask>",
            "activity": "<activity in one brief sentence>"
        }},
        ...,
        {{  
            "duration": "<duration in minutes>",
            "remaining_minutes": "<remaining minutes after completing this subtask>",
            "activity": "<activity in one brief sentence>"
        }}
    ]
}}
```

{identity}

Today is {today}. {task_context}

Task: Break down the task in subtasks 5 minute increments. At the end no time should be left.

In minimum 5 minutes increments, what are the subtasks that {name} does when {name} is "{task_description}" from {task_start_time} ~ {task_end_time}? (total duration in minutes: {task_duration}):
<|assistant|>```
{{
    "total_duration": "{task_duration}",
    "subtasks": [
        {{ 
            "duration": "5",
            "remaining_minutes": "55",
            "activity": \""""


class TaskDecomposition(BaseModel):

    name: str
    identity: str
    today: str
    task_context: str
    task_description: str
    task_start_time: str
    task_end_time: str
    task_duration: str

    async def run(self):
        _prompt = PromptTemplate(input_variables=["name",
                                                  "identity",
                                                  "today",
                                                  "task_context",
                                                  "task_description",
                                                  "task_start_time",
                                                  "task_end_time",
                                                  "task_duration"],
                                 template=_template)

        _task_decomposition_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 550,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.1}, verbose=global_state.verbose)
        
        schedules = []
        tasks = []
        for i in range(3):
            _task_decomposition_chain.llm_kwargs["cache_key"] = f"8task_decomposition_{self.name}_{global_state.tick}_{i}"
            tasks += [_task_decomposition_chain.arun(name=self.name,
                                                        identity=self.identity,
                                                        today=self.today,
                                                        task_context=self.task_context,
                                                        task_description=self.task_description,
                                                        task_start_time=self.task_start_time,
                                                        task_end_time=self.task_end_time,
                                                        task_duration=self.task_duration)]
        
        completions = await asyncio.gather(*tasks)
        for completion in completions:

            pattern = r'<\|assistant\|\>\n*```\n*(\{.*?\})\n```'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    subtasks = json_object["subtasks"]
                    
                    if subtasks[-1]["remaining_minutes"] != "0":
                        subtasks[-1]["duration"] = subtasks[-2]["remaining_minutes"]
                    
                    return [(task["activity"], int(task["duration"])) for task in subtasks]
                except:
                    pass
                
        print("Unable to decompose task.")
        return [("thinking about life", t) for t in range(0, int(self.task_duration), 5)]
