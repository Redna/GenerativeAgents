import dspy
from functools import lru_cache
from typing import Tuple, Union, Optional

# --- Signatures ---

class DecideToReactSignature(dspy.Signature):
    """Decide whether to react to an observation or wait, based on the context."""
    context: str = dspy.InputField(desc="The context of the situation.")
    current_time: str = dspy.InputField(desc="The current time.")
    agent: str = dspy.InputField(desc="The name of the agent deciding.")
    agent_with: str = dspy.InputField(desc="The name of the other agent involved.")
    agent_with_action: str = dspy.InputField(desc="What the other agent is doing.")
    agent_observation: str = dspy.InputField(desc="What the deciding agent is currently doing.")
    agent_with_observation: str = dspy.InputField(desc="Observation of the other agent.")
    initial_action_description: str = dspy.InputField(desc="The initial action the agent was doing.")
    thought_process: str = dspy.OutputField(desc="The reasoning behind the decision.")
    option: int = dspy.OutputField(desc="The chosen option (1 for Wait, 2 for Continue).")

class ActionEventTripleSignature(dspy.Signature):
    """Given a sentence identify the subject, predicate, and object from the sentence."""
    name: str = dspy.InputField(desc="Name of the agent.")
    action_description: str = dspy.InputField(desc="Description of action.")
    subject: str = dspy.OutputField(desc="The subject of the sentence (usually the name).")
    predicate: str = dspy.OutputField(desc="The action being performed.")
    object: str = dspy.OutputField(desc="The entity that the action is being performed on.")

class ObjectEventSignature(dspy.Signature):
    """Determine the state of an object that is being used by someone."""
    name: str = dspy.InputField(desc="Name of the agent.")
    object_name: str = dspy.InputField(desc="Name of the object.")
    action_description: str = dspy.InputField(desc="Description of the action being performed.")
    state: str = dspy.OutputField(desc="The new state of the object.")

class RatePoignanceSignature(dspy.Signature):
    """Rate the poignance (importance) of an event on a scale of 0 to 10."""
    agent_name: str = dspy.InputField(desc="The name of the agent.")
    agent_identity: str = dspy.InputField(desc="A description of the agent's identity and backstory.")
    event_type: str = dspy.InputField(desc="The type of the event (e.g., 'Exhibition', 'Chat').")
    description: str = dspy.InputField(desc="A description of the event.")
    rating: int = dspy.OutputField(desc="The rating of the poignance of the event (0-10).")

class ActionPronunciatioSignature(dspy.Signature):
    """Provide one or two emoji that best represents the following statement or emotion."""
    action_description: str = dspy.InputField(desc="Statement or emotion description.")
    emoji: str = dspy.OutputField(desc="Maximum two emojis.")

class ContextualizeEventSignature(dspy.Signature):
    """Write about the personality and observations of the agent based on a given event and related events."""
    agent: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Agent's identity context.")
    event_description: str = dspy.InputField(desc="Description of the perceived event.")
    events: str = dspy.InputField(desc="Related remembered events.")
    thoughts: str = dspy.InputField(desc="Agent's thoughts about the event.")
    event_context: str = dspy.OutputField(desc="Brief overview of things to remember for daily plan.")

# --- Modules ---

class ReactionDecider(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(DecideToReactSignature)
    
    def forward(self, context, current_time, agent, agent_with, agent_with_action,
                agent_observation, agent_with_observation, initial_action_description) -> int:
        try:
            response = self.predict(
                context=context, current_time=current_time, agent=agent,
                agent_with=agent_with, agent_with_action=agent_with_action,
                agent_observation=agent_observation, agent_with_observation=agent_with_observation,
                initial_action_description=initial_action_description
            )
            opt = response.option
            if opt not in [1, 2]: return 2
            return opt
        except Exception:
            return 2

class EventParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.triple = dspy.ChainOfThought(ActionEventTripleSignature)
        self.object_state = dspy.ChainOfThought(ObjectEventSignature)

    def get_triple(self, name: str, action_description: str, address: str = None) -> Tuple[str, str, str]:
        try:
            response = self.triple(name=name, action_description=action_description)
            subject = response.subject
            if address: subject = address
            return (subject, response.predicate, response.object)
        except Exception:
            return (address if address else name, "is", "doing something")

    def get_object_state(self, name: str, object_name: str, object_address: str, 
                        action_description: str) -> Tuple[str, Tuple[str, str, str]]:
        try:
            response = self.object_state(
                name=name, object_name=object_name, action_description=action_description
            )
            return f"{object_name} is {response.state}", (object_address, "is", response.state)
        except Exception:
            return f"{object_name} is in use", (object_address, "is", "in use")

class PoignanceRater(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(RatePoignanceSignature)

    def forward(self, agent_name: str, agent_identity: str, type_: str, description: str) -> int:
        try:
            response = self.predict(
                agent_name=agent_name, agent_identity=agent_identity,
                event_type=type_, description=description
            )
            return max(0, min(10, response.rating))
        except Exception:
            return 5

class EmojiMapper(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use Simple Predict if sufficient, Plan says Predict.
        self.predict = dspy.Predict(ActionPronunciatioSignature)

    def forward(self, action_description: str) -> str:
        try:
            response = self.predict(action_description=action_description)
            return response.emoji
        except Exception:
            return "😐"

class Contextualizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ContextualizeEventSignature)

    def forward(self, agent: str, identity: str, event_description: str, events: str, thoughts: str) -> str:
        try:
            response = self.predict(
                agent=agent, identity=identity, event_description=event_description,
                events=events, thoughts=thoughts
            )
            return response.event_context
        except Exception:
            return f"{agent} observed {event_description}."
