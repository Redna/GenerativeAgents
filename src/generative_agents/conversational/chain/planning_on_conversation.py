from pydantic import BaseModel
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are {{agent}}. You will write in the first person."""

user_shot_1 = """Conversation:
Alice Smith: Good morning, everyone. Let's start with the sales report.
Bob Johnson: Good morning! The sales in the last quarter increased by 15%.
Alice Smith: That's excellent news. Any updates on the marketing strategies?
Carol Taylor: Yes, the new campaign has been very successful, and we've seen a significant uptick in engagement on social media.
Alice Smith: Great! Let's keep the momentum going. Bob, please prepare a detailed report for the stakeholders.
Bob Johnson: Will do, Alice. I'll have it ready by tomorrow.
Alice Smith: Thank you, everyone. Let's wrap up for today.

What do you need to remember from the conversation? (it is for my planning and should be one full sentence)"""

ai_shot_1 = """I need to remember that sales increased by 15% in the last quarter and there has been a successful new marketing campaign leading to higher engagement on social media. A detailed report for stakeholders is needed by tomorrow, prepared by Bob Johnson."""

user = """Conversation:
{{conversation}}

What do you need to remember from the conversation? (it is for my planning and should be one full sentence)"""


class PlanningOnConversation(BaseModel):
    conversation: str
    agent: str

    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system, template_format="jinja2"),
                HumanMessagePromptTemplate.from_template(user_shot_1),
                AIMessagePromptTemplate.from_template(ai_shot_1),
                HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])
        _planning_on_conversation_chain = LLMChain(prompt=chat_template, llm=llm, verbose=True, llm_kwargs={
            "max_tokens": 80,
            "top_p": 0.90,
            "temperature": 0.4,
        })

        completion = await _planning_on_conversation_chain.ainvoke(input={"conversation":self.conversation, "agent":self.agent})
        return completion["text"]
    
async def __tests():
    t = [PlanningOnConversation(conversation="""Mike Brown: Hi, I'm having trouble accessing my account.
Nancy Green: Hi Mike, I'm here to help. Have you tried resetting your password?
Mike Brown: Yes, I tried that, but it didn't work.
Nancy Green: Okay, let me check your account details. Can you provide your account number?
Mike Brown: Sure, it's 123456789.
Nancy Green: Thanks, give me a moment to look into this.
...
Nancy Green: I've reset your account on our end. Please try logging in again.
Mike Brown: It works now. Thank you for your help, Nancy!
Nancy Green: You're welcome, Mike! Let me know if there's anything else I can do for you.""",
            agent="Nancy Green").run(),
            PlanningOnConversation(conversation="""Emily Jones: I've been using the new version of the app, and I've noticed a few issues.
David White: Oh? What kind of issues are you experiencing?
Emily Jones: The app crashes frequently, especially when I try to upload large files.
David White: That's concerning. We'll need to investigate that. Anything else?
Emily Jones: Yes, the new interface is a bit confusing. It took me a while to find the settings menu.
David White: I see. We appreciate your feedback, Emily. We'll look into these issues and work on improving the user experience.
Emily Jones: Thanks, David. I'm looking forward to the updates.""",
            agent="David White").run(),
            PlanningOnConversation(conversation="""Henry Larson: Hi Julia, I came across your recent work on renewable energy sources. It's quite impressive.
Julia Sanchez: Thank you, Henry! I'm glad you found it interesting.
Henry Larson: Absolutely. I'm working on a project related to solar energy, and I believe collaborating could be beneficial for both of us.
Julia Sanchez: That sounds like a great opportunity. What do you have in mind?
Henry Larson: I'm thinking about a joint research initiative that combines our expertise. We could tackle some of the bigger challenges in the field.
Julia Sanchez: I love the idea. Let's set up a meeting to discuss this further.
Henry Larson: Perfect. I'll send you an invitation for next week.
Julia Sanchez: Looking forward to it.""",
            agent="Henry Larson").run()]
    return await asyncio.gather(*t)

if __name__ == "__main__":
    import asyncio
    t = asyncio.run(__tests())
    print(t)
