"""
intelligence/modules/dialogue.py
Phase 6: Dialogue modules.

- TalkDecider REMOVED — conversation initiation is handled by ReActActor's `speak_to` tool.
  The LLM selects `speak_to(target, opening_line)` itself; no separate TalkDecider needed.

- DialogueGenerator, DialogueSummarizer, DialogueMemoer, DialoguePlanner kept for
  future use in turn-based chat generation (multi-turn conversation loop).
  These are NOT currently wired into the main agent brain — they are available for
  when the full chat turn loop is implemented.

- RelationshipSummarizer kept for reflection pipeline use.
"""
import dspy
from typing import Tuple


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

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
    end_conversation: bool = dspy.OutputField(desc="True if the conversation should end.")


class ConversationSummarySignature(dspy.Signature):
    """Summarize a conversation in one sentence."""
    conversation: str = dspy.InputField(desc="The conversation text.")
    summary: str = dspy.OutputField(desc="One sentence summary.")


class MemoOnConversationSignature(dspy.Signature):
    """Write a memo on what the agent found interesting from the conversation."""
    agent: str = dspy.InputField(desc="Name of the agent.")
    conversation: str = dspy.InputField(desc="The conversation text.")
    memo: str = dspy.OutputField(desc="One sentence memo.")


class PlanningOnConversationSignature(dspy.Signature):
    """Determine what to remember from a conversation (in first person)."""
    agent: str = dspy.InputField(desc="Name of the agent.")
    conversation: str = dspy.InputField(desc="The conversation text.")
    to_remember: str = dspy.OutputField(desc="One sentence on what to remember.")


class ChatRelationshipSignature(dspy.Signature):
    """Summarize what the agent knows about their relationship with another agent."""
    statements: str = dspy.InputField(desc="Statements about interactions.")
    agent: str = dspy.InputField(desc="Name of the agent.")
    agent_with: str = dspy.InputField(desc="Name of the other agent.")
    relationship_summary: str = dspy.OutputField(desc="Summary of the relationship.")


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class DialogueGenerator(dspy.Module):
    """
    Generates the next turn in an ongoing conversation.
    Used for multi-turn chat loop (not yet wired in the main brain).
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConversationSignature)

    def forward(self, agent, identity, memory, past_context, location,
                agent_action, agent_with, agent_with_action, conversation_history) -> Tuple[str, bool]:
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
    """Summarizes a completed conversation into one sentence."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConversationSummarySignature)

    def forward(self, conversation: str) -> str:
        try:
            return self.predict(conversation=conversation).summary
        except Exception:
            return "Conversation happened."


class DialogueMemoer(dspy.Module):
    """Extracts an agent's personal takeaway from a conversation."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(MemoOnConversationSignature)

    def forward(self, agent: str, conversation: str) -> str:
        try:
            return self.predict(agent=agent, conversation=conversation).memo
        except Exception:
            return "Nothing specific."


class DialoguePlanner(dspy.Module):
    """Extracts what the agent should remember from a conversation."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PlanningOnConversationSignature)

    def forward(self, agent: str, conversation: str) -> str:
        try:
            return self.predict(agent=agent, conversation=conversation).to_remember
        except Exception:
            return "I had a conversation."


class RelationshipSummarizer(dspy.Module):
    """Summarizes the relationship between two agents from memory statements."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ChatRelationshipSignature)

    def forward(self, statements: str, agent: str, agent_with: str) -> str:
        try:
            return self.predict(statements=statements, agent=agent, agent_with=agent_with).relationship_summary
        except Exception:
            return "They know each other."
