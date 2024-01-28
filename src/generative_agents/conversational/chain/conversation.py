import json
import re
from typing import Optional
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.chain.json_expert import JsonExpert
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.
You will act as the agent {agent}.

Your identity is: 
{identity}

<|user|>
Here is the memory that is in {agent}'s head:
{memory}

Past Context:
{past_context}

Current Location: {location}

Current Context:
{agent} was {agent_action} when {agent} saw {agent_with} in the middle of {agent_with_action}.
{agent} is initiating a conversation with {agent_with}.

{agent} and {agent_with} are chatting. Here is their conversation so far:
{conversation}

---
Output format: Output a valid json of the following format:
{{
    "{agent}": "<{agent}'s utterance>",
    "Reason if the conversation ended": "<reasoning for the conversation ending>",
    "Did the conversation end with {agent}'s utterance?": "<json Boolean>"
}}

Task: Given the above, what does {agent} say to {agent_with} next in the conversation? And did it end the conversation?
<|assistant|>
{{
    "{agent}": """

_prompt = PromptTemplate(input_variables=["identity",
                                                "memory", 
                                                "past_context", 
                                                "location",
                                                "agent", 
                                                "agent_action", 
                                                "agent_with", 
                                                "agent_with_action", 
                                                "conversation"],
                               template=_template)

_conversation_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                "max_new_tokens": 150, 
                                                "do_sample": True,
                                                "num_beams": 1,
                                                "temperature": 0.6,
                                                "top_p": 0.9,
                                                "top_k": 50},
                                                verbose=global_state.verbose)

class Conversation(BaseModel):
    identity: str
    memory: str
    past_context: Optional[str]
    location: str
    agent: str
    agent_action: str
    agent_with: str
    agent_with_action: str
    conversation: str

    async def run(self):  

        conversation_end_key = f"Did the conversation end with {self.agent}'s utterance?"
        utterance_key = self.agent

        for i in range(5):   
            _conversation_chain.llm_kwargs["cache_key"] = f"2conversation_chain_{self.agent}_{self.agent_with}_{global_state.tick}_{i}"

            completion = await _conversation_chain.arun(identity=self.identity,
                                                        memory=self.memory,
                                                        past_context=self.past_context,
                                                        location=self.location,
                                                        agent=self.agent,
                                                        agent_action=self.agent_action,
                                                        agent_with=self.agent_with,
                                                        agent_with_action=self.agent_with_action,
                                                        conversation=self.conversation)
            
            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                except Exception as error:
                    try:
                        json_object = await JsonExpert(wrong_json=match.group(1),
                                                    error_message=str(error)).run()
                    except:
                        continue

                try: 
                    return json_object[utterance_key], json_object[conversation_end_key]
                except:
                    pass
        
        print("Unable to generate the next utterance")
        return "I don't know what to say", True




