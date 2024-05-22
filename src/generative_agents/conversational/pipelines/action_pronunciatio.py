from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline


template = """You follow the tasks given by the user as close as possible.

Task: Provide one or two emoji that best represents the following statement or emotion: {{action_description}}"""

class Emoji(BaseModel):
    emoji: str = Field(
        description="Maximum two emojis that best represents the following statement or emotion.")

def action_pronunciatio(action_description: str) -> str:
    emoji = grammar_pipeline.run(model=Emoji, prompt_template=template, template_variables={
        "action_description": action_description
    })

    return emoji.emoji

if __name__ == "__main__":
    print(action_pronunciatio(action_description="Taking a shower"))
    print(action_pronunciatio(action_description="Drinking"))
    print(action_pronunciatio(action_description="Taking a bath"))
    print(action_pronunciatio(action_description="Visiting a friend"))
    print(action_pronunciatio(action_description="Walking around"))