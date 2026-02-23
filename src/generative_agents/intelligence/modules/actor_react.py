"""
intelligence/modules/actor_react.py
Phase 6: ReAct-style actor that replaces the 7+ sequential LLM calls in ActorLayer
with a single tool-choice call. The model picks one of 4 action tools;
Python dispatches it — zero additional LLM calls.
"""
import json
import dspy
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tool definitions (typed, no dspy.Tool dependency)
# ---------------------------------------------------------------------------

TOOL_NAMES = Literal["move_to", "speak_to", "wait", "update_action", "explore"]

TOOLS_DESCRIPTION = """Available tools:
1. move_to(action_description: str)  — PRIMARY tool for starting any new physical activity. Provide a natural language event description containing the action and location (e.g. 'Making breakfast in the kitchen' or 'Walking to the cafe'). The system will automatically pathfind you there.
2. speak_to(target_agent: str, opening_line: str) — Initiate dialogue with a nearby agent.
3. wait()                            — Do nothing. Stay in place and let your current action continue.
4. update_action(activity: str)      — ONLY use this if you are changing your state but STAYING EXACTLY WHERE YOU ARE (e.g. 'Reading a book on the current couch'). If your new activity requires interacting with a new object or room, use move_to instead!
5. explore(strategy: str) — Use this when you need to go somewhere but DO NOT know the location (e.g. if a move_to fails or you are lost). Strategies: 'wander randomly', 'search for [item/person]', or 'map area'.
"""


class ToolCall(BaseModel):
    tool: TOOL_NAMES = Field(description="Name of the tool to invoke.")
    args: dict = Field(
        description="Arguments for the tool as a flat key-value dict.",
        default_factory=dict,
    )


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------

class AgentStepSignature(dspy.Signature):
    """
    You are a generative agent deciding your next action for this simulation tick.

    Look at what you can see around you (visible_events) and your current daily plan.
    Choose ONE tool from the available tools and provide its arguments.
    Reason briefly before choosing.
    """

    name: str = dspy.InputField(desc="Agent's name.")
    identity: str = dspy.InputField(desc="Agent's identity, backstory, and traits.")
    current_plan: str = dspy.InputField(desc="The agent's current scheduled activity.")
    visible_events: str = dspy.InputField(
        desc="Numbered list of events the agent can perceive right now."
    )
    available_tools: str = dspy.InputField(
        desc="Description of the tools the agent can call."
    )
    tool_name: str = dspy.OutputField(
        desc="Exactly one tool name: move_to | speak_to | wait | update_action | explore"
    )
    tool_args: str = dspy.OutputField(
        desc='Tool arguments as a compact JSON object, e.g. {"destination": "library"}. Empty object {} for wait.'
    )


# ---------------------------------------------------------------------------
# DSPy Module
# ---------------------------------------------------------------------------

class ReActActor(dspy.Module):
    """
    Single LLM call that decides what the agent does next.
    Returns a validated ToolCall object for the ActorLayer to dispatch.
    """

    def __init__(self):
        super().__init__()
        # Use Predict — Qwen3-8B handles reasoning internally via <think> tokens.
        # Thinking is enabled at the LM level via fast_lm (default, thinking=False).
        # For tool selection we want fast, structured output, not deep reasoning.
        self.predict = dspy.Predict(AgentStepSignature)

    def forward(
        self,
        name: str,
        identity: str,
        current_plan: str,
        visible_events: list[str],
    ) -> ToolCall:
        events_str = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(visible_events)) or "Nothing notable nearby."

        try:
            result = self.predict(
                name=name,
                identity=identity,
                current_plan=current_plan,
                visible_events=events_str,
                available_tools=TOOLS_DESCRIPTION,
            )

            # Parse tool name (strip whitespace/quotes defensively)
            raw_tool = result.tool_name.strip().strip('"').strip("'").lower()
            # Map to valid literal
            valid_tools = {"move_to", "speak_to", "wait", "update_action", "explore"}
            tool_name = raw_tool if raw_tool in valid_tools else "wait"

            # Parse args JSON
            try:
                args = json.loads(result.tool_args or "{}")
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError):
                args = {}

            return ToolCall(tool=tool_name, args=args)  # type: ignore[arg-type]

        except Exception:
            return ToolCall(tool="wait", args={})
