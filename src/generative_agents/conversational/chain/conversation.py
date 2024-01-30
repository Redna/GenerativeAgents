import asyncio
import json
import re
from typing import Optional
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.chain.json_expert import JsonExpert
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate

system = """You will act as a person in a role-playing game. You are in a conversation with another person. You will be given a context and a conversation so far. You need to output valid JSON describing the next utterance and whether the conversation ended with your utterance.

Output format:
{{
    "utterance": "<utterance>",
    "endConversation": "<true or false>"
}}
"""

user_shot_1 = """You are John Doe. Your identity is: 
John Doe is a waiter at Hobbs Cafe. He is known for his excellent service.

Here is the memory that is in John Doe's head:
John Doe knows Giorgio Rossato for 5 years. They met at the Pharmacy.

Past Context:
John Doe was serving coffee to Giorgio Rossato last week.

Current Location: supermarket

Current Context:
John Doe was Buying milk for Hobbs Cafe. when John Doe saw Giorgio Rossato in the middle of Buying toilet paper for his family..
John Doe is initiating a conversation with Giorgio Rossato.

John Doe and Giorgio Rossato are chatting. Here is their conversation so far:


Given the context above, what does John Doe say to Giorgio Rossato next in the conversation? And did the conversation end?"""

ai_shot_1 = """{{
    "utterance": "Hey Giorgio, long time no see! How have you been since last week at the café? I'm here buying supplies for Hobbs Cafe today.",
    "endConversation": false
}}"""

user = """You are {agent}. Your identity is: 
{identity}

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

Given the context above, what does {agent} say to {agent_with} next in the conversation? And did it end the conversation?"""


chat_template = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(user_shot_1),
        AIMessagePromptTemplate.from_template(ai_shot_1),
        HumanMessagePromptTemplate.from_template(user)])


_conversation_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                                                "max_tokens": 200,
                                                "temperature": 0.6,
                                                "top_p": 0.9},
                                                verbose=True)

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
        for i in range(5):   
            completion = await _conversation_chain.ainvoke(input={"identity": self.identity,
                                                                "memory": self.memory,
                                                                "past_context": self.past_context,
                                                                "location": self.location,
                                                                "agent": self.agent,
                                                                "agent_action": self.agent_action,
                                                                "agent_with": self.agent_with,
                                                                "agent_with_action": self.agent_with_action,
                                                                "conversation": self.conversation})
            
            try:
                pattern = r'\{.*?\}'
                match = re.search(pattern, completion["text"], re.DOTALL)
                if match:
                    json_object = json.loads(match[0])
                
            except Exception as error:
                try:
                    json_object = await JsonExpert(wrong_json=match[0],
                                                error_message=str(error)).run()
                except:
                    continue

            try: 
                return json_object["utterance"], json_object["endConversation"]
            except:
                pass
        
        print("Unable to generate the next utterance")
        return "I don't know what to say", True


async def __tests():
    global_state.verbose = True
    t = [Conversation(identity="Alice Smith is a librarian at the City Library. She is appreciated for her helpful nature.",
            memory="Alice Smith has known Emily Johnson for 3 years. They met at a book club.",
            past_context="Alice Smith recommended a mystery novel to Emily Johnson last month.",
            location="bookstore",
            agent="Alice Smith",
            agent_action="Searching for new books for the library.",
            agent_with="Emily Johnson",
            agent_with_action="Looking for a birthday gift for her nephew.",
            conversation="").run(),
        Conversation(identity="Mark Turner is a gym instructor at FitLife Gym. He is popular for his motivational coaching.",
            memory="Mark Turner has been friends with David Lee for 2 years. They met during a fitness workshop.",
            past_context="Mark Turner was helping David Lee with a new workout routine yesterday.",
            location="sports store",
            agent="Mark Turner",
            agent_action="Buying new gym equipment for FitLife Gym.",
            agent_with="David Lee",
            agent_with_action="Choosing running shoes for his marathon training.",
            conversation="").run(),
        Conversation(identity="Sarah Johnson is a chef at The Gourmet Kitchen. She is famous for her innovative recipes.",
            memory="Sarah Johnson knows Chloe Adams for 6 months. They met at a cooking class.",
            past_context="Sarah Johnson was teaching Chloe Adams how to bake French pastries last Saturday.",
            location="farmer's market",
            agent="Sarah Johnson",
            agent_action="Selecting fresh produce for her restaurant.",
            agent_with="Chloe Adams",
            agent_with_action="Buying organic herbs for her home garden.",
            conversation="").run(),
        Conversation(identity="Kevin Brown is a software developer at Tech Innovations. He is known for his problem-solving skills.",
            memory="Kevin Brown has been colleagues with Rachel Green for 4 years. They met at a tech conference.",
            past_context="Kevin Brown was discussing a new project with Rachel Green two days ago.",
            location="coffee shop",
            agent="Kevin Brown",
            agent_action="Working on a software update on his laptop.",
            agent_with="Rachel Green",
            agent_with_action="Preparing a presentation for their next client meeting.",
            conversation="").run(),

        Conversation(identity="Emma Wilson is a school teacher at Sunnydale Elementary. She is loved for her patience and creativity.",
            memory="Emma Wilson has known Laura Martinez for 8 years. They met at a teacher training seminar.",
            past_context="Emma Wilson was helping Laura Martinez with classroom decorations last Friday.",
            location="craft store",
            agent="Emma Wilson",
            agent_action="Buying art supplies for her students.",
            agent_with="Laura Martinez",
            agent_with_action="Looking for educational games for her classroom.",
            conversation="").run()
    ]
    
    return await asyncio.gather(*t)

if __name__ == "__main__":
    t = asyncio.run(__tests())
    print(t)

