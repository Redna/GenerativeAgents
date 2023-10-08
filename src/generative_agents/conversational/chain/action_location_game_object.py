from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Current activity: sleep in bed
Objects available: [bed, easel, closet, painting]
Pick ONE most relevant object from the objects available: bed
---
Current activity: painting
Objects available: [easel, closet, sink, microwave]
Pick ONE most relevant object from the objects available: easel
---
Current activity: cooking
Objects available: [stove, sink, fridge, counter]
Pick ONE most relevant object from the objects available: stove
---
Current activity: watch TV
Objects available: [couch, TV, remote, coffee table]
Pick ONE most relevant object from the objects available: TV
---
Current activity: study
Objects available: [desk, computer, chair, bookshelf]
Pick ONE most relevant object from the objects available: desk
---
Current activity: talk on the phone
Objects available: [phone, charger, bed, nightstand]
Pick ONE most relevant object from the objects available: phone
---
Current activity: {action_description}
Objects available: [{objects_available}]
Pick ONE most relevant object from the objects available:"""

_prompt = PromptTemplate(input_variables=["action_description",
                                          "objects_available"],
                            template=_template)

action_location_game_object_chain = LLMChain(prompt=_prompt, llm=llm)
