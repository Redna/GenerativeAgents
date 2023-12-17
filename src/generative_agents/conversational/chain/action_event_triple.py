import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.

<|user|>
Output format: Output a valid json of the following format:
{{
    "sentence": "<sentence to extract the subject, predicate, and object from>",
    "subject": "<subject usually appears before the verb. For example, in 'The cat sleeps,' 'the cat' is the subject.>",
    "predicate": "<predicate in a sentence is everything that describes what the subject is doing or being, including the verb and often additional information like objects or adverbs. For example, in 'Birds sings a song,' 'sing' is the predicate.>",
    "object": "<object in a sentence is the noun or pronoun that receives the action of the verb, as in 'She threw the ball,' where 'the ball' is the object.>"
}}

Task: Given a sentence, identify the subject, predicate, and object from the sentence.
{{
    "sentence": "Dolores Murphy is eating breakfast.",
    "subject": "Dolores Murphy",
    "predicate": "eat",
    "object": "breakfast"
}}
--- 
Task: Given a sentence, identify the subject, predicate, and object from the sentence.
{{
    "sentence": "Michael Bernstein is writing email on a computer",
    "subject": "Michael Bernstein",
    "predicate": "write",
    "object": "email"
}}
---

Task: Given a sentence, identify the subject, predicate, and object from the sentence.
<|assistant|>
{{
    "sentence": "{name} is {action_description}",
    "subject": "{name}",
    "predicate": \""""


class ActionEventTriple(BaseModel):
    name: str
    address: str = None
    action_description: str

    async def run(self): 
        _prompt = PromptTemplate(input_variables=["name",
                                                  "action_description"],
                                      template=_template)

        for i in range(5):
            _action_event_triple_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                              "max_new_tokens":25,
                                                              "do_sample": True,
                                                              "top_p": 0.95,
                                                              "top_k": 10,
                                                              "temperature": 0.4,
                                                              "cache_key": f"9action_event_triple_chain_{self.name}_{self.action_description}_{i}_{global_state.tick}"},
                                                              verbose=global_state.verbose)

            completion = await _action_event_triple_chain.arun(name=self.name, action_description=self.action_description)
            
            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))

                    subject = json_object["subject"] if not self.address else self.address
                    return (subject, json_object["predicate"], json_object["object"])
                except:
                    pass 
        
        print("Unable to generate action event triple.")
        return self.address or self.name, self.action_description, "idle"
