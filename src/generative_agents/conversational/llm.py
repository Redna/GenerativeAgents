import asyncio

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key="na", openai_api_base="http://localhost:8080/v1/")


async def __run():
    tasks = [llm.ainvoke("Tell me a joke."), llm.ainvoke("Tell me a good joke."), llm.ainvoke("Tell me a bad joke.")]
    return await asyncio.gather(*tasks)

async def __run_chain():

    _template = """<|system|> You write a concise description about {agent}'s personality, family situation and characteristics. You include ALL the details provided in the given context (you MUST include all the names of persons, ages,...)."""
    _prompt = PromptTemplate(input_variables=["agent", "identity"],
                            template=_template)
    
    _identity_chain = LLMChain(prompt=_prompt, llm=llm,
    verbose=True)

    x= await _identity_chain.arun(agent="agent", identity="identity")
    print(x)


async def __runall():
    r = await __run()
    print(r)
    await __run_chain()

if __name__ == "__main__":
    asyncio.run(__runall())
    pass