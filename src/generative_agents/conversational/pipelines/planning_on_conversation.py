from enum import Enum
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You are {{agent}}. You will write in the first person.
Conversation:
{{conversation}}

What do you need to remember from the conversation? (it is for my planning and should be one full sentence)"""


class PlanningOnConversation(BaseModel):
    to_remember: str = Field(
        description="One full sentence on what to remember from the conversation.")


def planning_on_conversation(agent: str, conversation: str) -> str:
    planning_on_conversation = grammar_pipeline.run(model=PlanningOnConversation, prompt_template=template, template_variables={
        "agent": agent,
        "conversation": conversation
    })

    return planning_on_conversation.to_remember


if __name__ == "__main__":
    planning_on_conversation(agent="Alice Smith", conversation="""Alice Smith: Good morning, everyone. Let's start with the sales report.
                                 Bob Johnson: Good morning! The sales in the last quarter increased by 15%.
                                 Alice Smith: That's excellent news. Any updates on the marketing strategies?
                                 Carol Taylor: Yes, the new campaign has been very successful, and we've seen a significant uptick in engagement on social media.
                                 Alice Smith: Great! Let's keep the momentum going. Bob, please prepare a detailed report for the stakeholders.
                                 Bob Johnson: Will do, Alice. I'll have it ready by tomorrow.
                                 Alice Smith: Thank you, everyone. Let's wrap up for today.""")

    planning_on_conversation(agent="Nancy Green", conversation="""Mike Brown: Hi, I'm having trouble accessing my account.
                                    Nancy Green: Hi Mike, I'm here to help. Have you tried resetting your password?
                                    Mike Brown: Yes, I tried that, but it didn't work.
                                    Nancy Green: Okay, let me check your account details. Can you provide your account number?
                                    Mike Brown: Sure, it's 123456789.
                                    Nancy Green: Thanks, give me a moment to look into this.
                                    ...
                                    Nancy Green: I've reset your account on our end. Please try logging in again.
                                    Mike Brown: It works now. Thank you for your help, Nancy!
                                    Nancy Green: You're welcome, Mike! Let me know if there's anything else I can do for you.""")

    planning_on_conversation(agent="David White", conversation="""Emily Jones: I've been using the new version of the app, and I've noticed a few issues.
                                        David White: Oh? What kind of issues are you experiencing?
                                        Emily Jones: The app crashes frequently, especially when I try to upload large files.
                                        David White: That's concerning. We'll need to investigate that. Anything else?
                                        Emily Jones: Yes, the new interface is a bit confusing. It took me a while to find the settings menu.
                                        David White: I see. We appreciate your feedback, Emily. We'll look into these issues and work on improving the user experience.
                                        Emily Jones: Thanks, David. I'm looking forward to the updates.""")

    planning_on_conversation(agent="Henry Larson", conversation="""Henry Larson: Hi Julia, I came across your recent work on renewable energy sources. It's quite impressive.
                                        Julia Sanchez: Thank you, Henry! I'm glad you found it interesting.
                                        Henry Larson: Absolutely. I'm working on a project related to solar energy, and I believe collaborating could be beneficial for both of us.
                                        Julia Sanchez: That sounds like a great opportunity. What do you have in mind?
                                        Henry Larson: I'm thinking about a joint research initiative that combines our expertise. We could tackle some of the bigger challenges in the field.
                                        Julia Sanchez: I love the idea. Let's set up a meeting to discuss this further.
                                        Henry Larson: Perfect. I'll send you an invitation for next week.
                                        Julia Sanchez: Looking forward to it.""")
