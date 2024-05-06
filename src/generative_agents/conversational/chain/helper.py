from enum import Enum
from typing import Any, List

from generative_agents.conversational.llm import tokenizer

from langchain.prompts import PromptTemplate

class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"

class ChatMessage():
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {"role": self.role.value, "content": self.content}

class SystemMessage(ChatMessage):
    def __init__(self, content: str):
        super().__init__(Role.SYSTEM, content)

class AIMessage(ChatMessage):
    def __init__(self, content: str):
        super().__init__(Role.ASSISTANT, content)

class UserMessage(ChatMessage):
    def __init__(self, content: str):
        super().__init__(Role.USER, content)

class ChatTemplate():
    def __init__(self, messages: List[ChatMessage] = None):
        if messages is None:
            messages = []
        self.messages = messages
    
    def add_ai_message(self, message):
        self.messages.append({"role": Role.ASSISTANT.value, "content": message})
    
    def add_user_message(self, message):
        self.messages.append({"role": Role.USER.value, "content": message})
    
    def add_system_message(self, message):
        self.messages.append({"role": Role.SYSTEM.value, "content": message})
    
    def chat_messages(self):
        return [message.to_dict() for message in self.messages]
    
    def get_prompt_template(self, format="jinja2"):
        resolved = tokenizer.apply_chat_template(self.chat_messages(), tokenize=False)
        return PromptTemplate.from_template(resolved, format=format)
