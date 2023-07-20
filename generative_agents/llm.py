import re
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import torch


rate_information_importance = PromptTemplate(
                        input_variables=["memory"],
                        template="""
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.
Memory: {memory}
Rating:""".strip()
)

long_term_plan = PromptTemplate(
                        input_variables=["name", "identity", "known_locations", "recent_experiences", "date_string"],
                        template="""
{identity}
{recent_experiences}

* Only use the available locations: [{known_locations}]
* Include the time {name} wake's up and falls asleep

Example of a plan: 
8:00 am at the Oak Hill College Dorm: Klaus Mueller's room: bed, wake up and get ready for school;
9:00 am at the Oak Hill College Dorm: Klaus Mueller's room: desk, read and take notes for research paper;
...
5:00 pm at the Oak Hill College Dorm: Kitchen, get ready for dinner
10:00 am at the Oak Hill College Dorm: Klaus Mueller's room: bathroom, get ready for bed

Today is {date_string}. Here is {name}'s plan today in broad strokes:
""")

short_term_plan = PromptTemplate(
                        input_variables=["name", "identity", "daily_plan", "recent_experiences", "time_string", "date_string"],
                        template="""
{identity}
Daily plan: 
{daily_plan}
Recent Experiences: 
{recent_experiences}

* Give a duration for every plan

Break down {name}'s plan for the next hour starting at the current time.

The current time is {time_string}

Today is {date_string}. Here is {name}'s plan today in broad strokes:
""")

update_short_term_plan_question = PromptTemplate(
                        input_variables=["name", "identity", "short_term_plan", "recent_experiences", "observation"],
                        template="""
{identity}
Short term plan: {short_term_plan}
Recent experiences: {recent_experiences}
Observation: {observation}

Should {name} update their short term plan?
1: Update
2: Don't update

Number:""")

update_long_term_plan_question = PromptTemplate(
                        input_variables=["name", "identity", "daily_plan", "recent_experiences", "observation"],
                        template="""
{identity}
Daily plan: {daily_plan}
Recent experiences: {recent_experiences}
Observation: {observation}

Should {name} update their daily plan?
1: Update
2: Don't update

Number:""")

class LLM:
    def __init__(self):
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = "meta-llama/Llama-2-7b-chat-hf"
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_new_tokens=250,
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        self.chains = {
            "rate_information_importance": LLMChain(llm=llm, prompt=rate_information_importance),
            "long_term_plan": LLMChain(llm=llm, prompt=long_term_plan),
            "short_term_plan": LLMChain(llm=llm, prompt=short_term_plan),
            "update_short_term_plan_question": LLMChain(llm=llm, prompt=update_short_term_plan_question),
            "update_long_term_plan_question": LLMChain(llm=llm, prompt=update_long_term_plan_question)
        }

    def rate_information_importance(self, information) -> int:

        # parse a string e.g. ' 6' to an integer and catch the exception
        rating = self.chains["rate_information_importance"].run(information)

        # check if the rating is a number and if it is between 1 and 10
        match = re.compile(r"\d{1,2}").search(rating)

        if match:
            return int(match.group())

        print(f"unable not parse rating: {rating}")
        return 5

    def long_term_plan(self, name, identity, known_locations, recent_experiences, date_string) -> str:
        plan = self.chains["long_term_plan"].run({
            "name": name,
            "identity": identity,
            "known_locations": known_locations,
            "recent_experiences": recent_experiences,
            "date_string": date_string
        })
        return plan
    
    def short_term_plan(self, name, identity, daily_plan, recent_experiences, time_string, date_string) -> str:
        plan = self.chains["short_term_plan"].run({
            "name": name,
            "identity": identity,
            "daily_plan": daily_plan,
            "known_locations": known_locations,
            "recent_experiences": recent_experiences,
            "time_string": time_string,
            "date_string": date_string
        })
        return plan
    
    def should_update_short_term_plan(self, name, identity, short_term_plan, recent_experiences, observation) -> bool:
        should_update = self.chains["update_short_term_plan_question"].run({
            "name": name,
            "identity": identity,
            "short_term_plan": short_term_plan,
            "recent_experiences": recent_experiences,
            "observation": observation
        })
        return should_update == "1"
    
    def should_update_long_term_plan(self, name, identity, daily_plan, recent_experiences, observation) -> bool:
        should_update = self.chains["update_long_term_plan_question"].run({
            "name": name,
            "identity": identity,
            "daily_plan": daily_plan,
            "recent_experiences": recent_experiences,
            "observation": observation
        })
        return should_update == "1"
        
llm = LLM()