
from datetime import datetime
import re
from typing import List, Tuple

from langchain import PromptTemplate
import torch


JOHN_LIN = "John Lin"
EDDY_LIN = "Eddy Lin"


def get_timestamp() -> str:
    return f"It is {datetime.now().strftime('%B %d, %Y, %I:%M%p.')}"


class Agent:

    def __init__(self, name, summary, status):
        self.name = name
        self.initial_summary = summary
        self.initial_status = status
        self.memory = Memory()

    @property
    def status(self):
        return self.initial_status

    @property
    def summary(self):
        return self.initial_summary

class Dialogue:
    dialogue_init_template = PromptTemplate(
        input_variables=["timestamp", "name", "summary", "status", "observation", "relevant_memory", "action", "other_name"],
        template="""
{summary}

{timestamp}
{name}'s status: {status}
Observation: {observation}
Summary of relevant context from {name}'s memory:
{relevant_memory}

{action}

What would {name} say to {other_name}?
""".strip()
                )

    dialogue_cont_template = PromptTemplate(
        input_variables=["timestamp", "name", "summary", "status", "observation", "relevant_memory", "other_name", "dialogue_history"],
        template="""
{summary}

{timestamp}
{name}'s status: {status}
Observation: {observation}
Summary of relevant context from {name}'s memory:
{relevant_memory}

Here is the conversation history:
{dialogue_history}

What would {name} respond to {other_name}?
""".strip())


    def __init__(self, tokenizer, pipeline):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.dialogue_history = []

    def start(self, agent: Agent, other_agent: Agent, observation: str, relevant_memory: str, action: str):
        # parse datetime.now() in "February 13, 2023, 4:56pm."
        timestamp = f"It is {datetime.now().strftime('%B %d, %Y, %I:%M%p.')}"
    
        return self._converse(agent.name, self.dialogue_init_template.format(timestamp=timestamp, 
                    name=agent.name, 
                    summary=agent.summary,
                    relevant_memory=relevant_memory, 
                    status=agent.status, 
                    observation=observation, 
                    action=action, 
                    other_name=other_agent.name))

    def turn(self, agent: Agent, other_agent: Agent, observation: str, relevant_memory: str, dialogue_history: List[str]):
        # parse datetime.now() in "February 13, 2023, 4:56pm."
        timestamp = f"It is {datetime.now().strftime('%B %d, %Y, %I:%M%p.')}"

        dialogue_history_string = '\n- '.join(dialogue_history)
    
        return self._converse(agent.name, self.dialogue_cont_template.format(timestamp=timestamp, 
                    name=agent.name, 
                    summary=agent.summary,
                    relevant_memory=relevant_memory, 
                    status=agent.status, 
                    observation=observation, 
                    other_name=other_agent.name, 
                    dialogue_history=dialogue_history_string))
    
    def _converse(self, name, prompt: str) -> List[Tuple[str, str]]:
        prompt_len = len(self.tokenizer.encode(prompt))

        with torch.no_grad():
            out = self.pipeline(prompt, 
                            max_length=prompt_len + 75,
                            do_sample=True,
                            top_k=0,
                            top_p=.85,
                            temperature=.4,
                            streamer = None,
                            pad_token_id = self.tokenizer.eos_token_id
                            )
        
        dialog = out[0]['generated_text'][len(prompt):]

        print(dialog)

        statement = dialog.split("\n\n")[0] 
        statement = statement.replace('"', '')
        statement = statement.split(": ")[-1]
        statement = re.sub(".*would.*, ", "", statement) if "would" in statement[:20] else statement
        statement = statement[1:] if statement[0] == "\n" else statement
        statement = "\n".join(list(set(statement.split('\n'))))
        statement = statement[1:] if statement[0] == "\n" else statement

        if statement[-1] not in ['.', '?', '!']:
            print(re.search(".*[\.\?\!]", statement).end())
            statement = statement[:re.search(".*[\.\?\!]", statement).end()]
        
        direct_speech = f'{name}: "{statement}"'
        
        return direct_speech, dialog, out[0]['generated_text']

        


class Action:
    action_template = PromptTemplate(
                        input_variables=["timestamp", "name", "summary", "status", "observation", "relevant_memory"],
                        template="""
{summary}

{timestamp}
{name}'s status: {status}

Observation: {observation}

Summary of relevant context from {name}'s memory:
{relevant_memory}

Should {name} react to the observation, and if so, what would be an appropriate reaction?""".strip()
)

    def __init__(self, tokenizer, pipeline):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
    
    def generate(self, agent: Agent, observation: str, relevant_memory: str):
        action_prompt = self.action_template.format(
            timestamp = get_timestamp(),
            summary = agent.summary,
            name = agent.name,
            status = agent.status,
            observation = observation,
            relevant_memory = relevant_memory)
        
        prompt_len = len(self.tokenizer.encode(action_prompt))

        with torch.no_grad():
            out = self.pipeline(action_prompt, 
                            max_length=prompt_len + 80,
                            do_sample=True,
                            top_k=0,
                            top_p=.90,
                            temperature=.9,
                            streamer = None,
                            pad_token_id = self.tokenizer.eos_token_id
                            )
        
        print(out[0]['generated_text'])

        action = out[0]['generated_text'][len(action_prompt):]

        if action[-1] not in ['.', '?', '!']:
            statement = action[:re.search(".*[\.\?\!]", action).end()]
        else:
            statement = action

        return statement, action, out[0]['generated_text']


def __fallback(self, action):
    action_prompt = f"""
{action}

Reformulate the text above to one specific action in perfect tense:""".strip()
        
    with torch.no_grad():
        action = self.pipeline(action_prompt, 
                        max_length=len(self.tokenizer.encode(action_prompt)) + 30,
                        do_sample=True,
                        top_k=0,
                        top_p=.80,
                        temperature=.7,
                        streamer = None,
                        pad_token_id = self.tokenizer.eos_token_id
                        )

    print("*" * 50)

    print(action[0]['generated_text'])

    action = action[0]['generated_text'][len(action_prompt):]

    action = re.sub(r"could (\w+)", r"is \g<1>ing", action)
    action = re.sub(r"could't (\w+)", r"is not \g<1>ing", action)
    action = re.sub(r"should (\w+)", r"is \g<1>ing", action)
    action = re.sub(r"shouldn't (\w+)", r"is not \g<1>ing", action)
    action = re.sub(r"would (\w+)", r"is \g<1>ing", action)
    action = re.sub(r"wouldn't (\w+)", r"is not \g<1>ing", action)

    print("*" * 50)
    return None

class Memory:

    def __init__(self):
        self.memory = []
    
    def memory_for(self, agent):
        if agent.name == JOHN_LIN:
            return """Eddy Lin is John Lin's son. Eddy Lin has been working on music composition for his class. 
Eddy Lin likes to walk around the garden when he is thinking about or listening to music."""
        else:
            return """John Lin is Eddy Lin’s father. John Lin is caring and is interested to learn more about Eddy 
Lin’s school work. John Lin knows that Eddy Lin is working on a music composition."""
        



    


    

