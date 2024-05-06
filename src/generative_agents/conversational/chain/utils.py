from typing import Dict
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

NEWLINE = "\n"

def merge_ai_opening_with_completion(chat_template: ChatPromptTemplate, inputs: Dict[str, any], completion: str):
    messages = chat_template.format_messages(**inputs)

    ai_opening = messages[-1].content.strip()

    last_text = ai_opening.split(NEWLINE)[-1].strip()

    if last_text in completion:
        index = completion.index(last_text)
        completion = completion[index + len(last_text):]

    return ai_opening + " " + completion.strip()