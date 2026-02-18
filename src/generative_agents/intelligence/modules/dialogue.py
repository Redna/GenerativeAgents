import dspy
from typing import Tuple

# --- Signatures ---

class DecideToTalkSignature(dspy.Signature):
    """Decide whether to initiate a conversation based on the context, current time, and observations."""
    context: str = dspy.InputField(desc="The context of the situation.")
    current_time: str = dspy.InputField(desc="The current time.")
    init_agent: str = dspy.InputField(desc="The name of the agent deciding to talk.")
    agent_with: str = dspy.InputField(desc="The name of the agent to potentially talk to.")
    last_chat_summary: str = dspy.InputField(desc="Summary of the last conversation between the agents.")
    init_agent_observation: str = dspy.InputField(desc="What the initiating agent is currently doing.")
    agent_with_observation: str = dspy.InputField(desc="What the other agent is currently doing.")
    thought_process: str = dspy.OutputField(desc="The reasoning behind the decision.")
    initiate_conversation: bool = dspy.OutputField(desc="True if the agent decides to initiate a conversation, False otherwise.")

class ConversationSignature(dspy.Signature):
    """Generate the next utterance in a conversation and decide if it ends."""
    agent: str = dspy.InputField(desc="Name of the acting agent.")
    identity: str = dspy.InputField(desc="Identity of the acting agent.")
    memory: str = dspy.InputField(desc="Relevant memory of the agent.")
    past_context: str = dspy.InputField(desc="Context from past interactions.")
    location: str = dspy.InputField(desc="Current location.")
    agent_action: str = dspy.InputField(desc="Current action of the agent.")
    agent_with: str = dspy.InputField(desc="The person the agent is talking to.")
    agent_with_action: str = dspy.InputField(desc="Action of the other person.")
    conversation_history: str = dspy.InputField(desc="Conversation so far.")
    utterance: str = dspy.OutputField(desc="The next utterance.")
    end_conversation: bool = dspy.OutputField(desc="True if the conversation should end, False otherwise.")

class ConversationSummarySignature(dspy.Signature):
    """Summarize a conversation in one sentence."""
    conversation: str = dspy.InputField(desc="The conversation text.")
    summary: str = dspy.OutputField(desc="One sentence summary.")

class MemoOnConversationSignature(dspy.Signature):
    """Write a memo on what the agent found interesting from the conversation."""
    agent: str = dspy.InputField(desc="Name of the agent.")
    conversation: str = dspy.InputField(desc="The conversation text.")
    memo: str = dspy.OutputField(desc="One sentence memo of what was interesting.")

class PlanningOnConversationSignature(dspy.Signature):
    """Determine what to remember from a conversation (in first person)."""
    agent: str = dspy.InputField(desc="Name of the agent.")
    conversation: str = dspy.InputField(desc="The conversation text.")
    to_remember: str = dspy.OutputField(desc="One sentence on what to remember.")

class ChatRelationshipSignature(dspy.Signature):
    """Summarize what the agent feels or knows about their relationship with another agent based on statements."""
    statements: str = dspy.InputField(desc="Statements about interactions.")
    agent: str = dspy.InputField(desc="Name of the agent.")
    agent_with: str = dspy.InputField(desc="Name of the other agent.")
    relationship_summary: str = dspy.OutputField(desc="Summary of the relationship.")

# --- Modules ---

class TalkDecider(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(DecideToTalkSignature)

    def forward(self, context, current_time, init_agent, agent_with, last_chat_summary, 
                init_agent_observation, agent_with_observation) -> bool:
        try:
            response = self.predict(
                context=context, current_time=current_time, init_agent=init_agent,
                agent_with=agent_with, last_chat_summary=last_chat_summary,
                init_agent_observation=init_agent_observation, agent_with_observation=agent_with_observation
            )
            return response.initiate_conversation
        except Exception:
            return False

class DialogueGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ConversationSignature)

    def forward(self, agent, identity, memory, past_context, location, agent_action,
                agent_with, agent_with_action, conversation_history) -> Tuple[str, bool]:
        try:
            response = self.predict(
                agent=agent, identity=identity, memory=memory, past_context=past_context,
                location=location, agent_action=agent_action, agent_with=agent_with,
                agent_with_action=agent_with_action, conversation_history=conversation_history
            )
            return response.utterance, response.end_conversation
        except Exception:
            return "...", True

class DialogueSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ConversationSummarySignature)

    def forward(self, conversation: str) -> str:
        try:
            return self.predict(conversation=conversation).summary
        except Exception:
            return "Conversation happened."

class DialogueMemoer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(MemoOnConversationSignature)

    def forward(self, agent: str, conversation: str) -> str:
        try:
            return self.predict(agent=agent, conversation=conversation).memo
        except Exception:
            return "Nothing specific."

class DialoguePlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PlanningOnConversationSignature)

    def forward(self, agent: str, conversation: str) -> str:
        try:
            return self.predict(agent=agent, conversation=conversation).to_remember
        except Exception:
            return "I had a conversation."

class RelationshipSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ChatRelationshipSignature)

    def forward(self, statements: str, agent: str, agent_with: str) -> str:
        try:
            return self.predict(statements=statements, agent=agent, agent_with=agent_with).relationship_summary
        except Exception:
            return "They know each other."
