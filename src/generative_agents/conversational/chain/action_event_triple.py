from dataclasses import Field
import json

from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.chain.helper import AIMessage, ChatMessage, ChatTemplate, SystemMessage, UserMessage
from generative_agents.conversational.chain.utils import merge_ai_opening_with_completion
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain

class ActionEvent(BaseModel):
    subject: str = Field(description="The subject is the entity performing the action")
    predicate: str = Field(description="The predicate is the action being performed")
    object: str = Field(description="The object is the entity that the action is being performed on")

prompt = """You follow the tasks given by the user as close as possible. You need to return a valid JSON.

Task: Given a sentence identify the subject, predicate, and object from the sentence.
Sentence: {name} is {action_description}"""



class ActionEventTriple(BaseModel):
    name: str
    address: str = None
    action_description: str

    async def run(self):

        chat_template = ChatTemplate(messages = [UserMessage(prompt)])
        
        _action_event_triple_chain = LLMChain(prompt=chat_template.get_prompt_template(format="f-string"), 
                                                  llm=llm, llm_kwargs={
                                                                "max_tokens": 45,
                                                                "top_p": 0.95,
                                                                "temperature": 0.4},
                                                                verbose=global_state.verbose)
        



            inputs = {"name": self.name, "action_description": self.action_description}
            completion = await _action_event_triple_chain.ainvoke(input=inputs)

            full_completion = completion["text"]
            try:
                json_object = json.loads(full_completion)

                subject = json_object["subject"] if not self.address else self.address

                object_ = json_object["object"] if type(
                    json_object["object"]) == str else json_object["object"][0]
                return (subject, json_object["predicate"], object_)
            except:
                pass

        print("Unable to generate action event triple.")
        return self.address or self.name, self.action_description, "idle"


if __name__ == "__main__":
    import asyncio
    t = asyncio.run(ActionEventTriple(
        name="John Doe", action_description="John Doe is taking a warm shower").run())
    print(t)
