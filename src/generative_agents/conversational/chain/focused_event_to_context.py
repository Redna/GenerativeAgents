import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Task -- given the context below, write a summary about {agent}'s personality and {agent}'s observations.

### Context
{identity}
{agent} perceived the following event: {event_description}
He remembered the following related events: {events}
He thought the following about the event: {thoughts}

### Summary:"""

_prompt = PromptTemplate(input_variables=["agent",
                                          "identity",
                                          "event_description",
                                          "events",
                                          "thoughts"],
                         template=_template)

class FocusedEventToContext(BaseModel):
    agent: str
    identity: str
    event_description: str
    events: str
    thoughts: str

    async def run(self):

        _focused_event_to_context = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 200,
            "do_sample": True,
            "top_p": 0.92,
            "top_k": 50,
            "temperature": 0.8,
            "cache_key": f"2focused_event_to_context_{self.agent}_{global_state.tick}"}, verbose=global_state.verbose)

        completion = await _focused_event_to_context.arun(identity=self.identity,
                                                          agent=self.agent,
                                                          event_description=self.event_description,
                                                          events=self.events,
                                                          thoughts=self.thoughts)
        pattern = r"Summary:[\n ]{0,3}(.*)"
        return re.findall(pattern, completion)[-1]