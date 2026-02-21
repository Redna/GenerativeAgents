"""
intelligence/modules/retrieval.py
Phase 6: ContextualReranker — a DSPy module that re-ranks candidate memories
relative to the agent's current situation. One batched LLM call replaces N
per-event poignancy calls.
"""
import dspy
from typing import Any

from generative_agents.common.dspy_config import thinking_lm


class ContextualRerankSignature(dspy.Signature):
    """
    Given what the agent is doing right now and their active plan,
    rank the candidate memories from most to least relevant for making a
    good decision THIS tick. Ignore memories that are clearly unrelated.
    """

    agent_name: str = dspy.InputField(desc="The agent's name.")
    current_situation: str = dspy.InputField(
        desc="What the agent is currently doing or perceiving."
    )
    current_plan: str = dspy.InputField(desc="The agent's active scheduled activity.")
    candidate_memories: str = dspy.InputField(
        desc="Numbered list of memories with their abstraction level, e.g. '0. [L0] Klaus read a book'."
    )
    ranked_indices: list[int] = dspy.OutputField(
        desc="List of candidate indices (0-based) ordered from most to least important right now."
    )


class ContextualReranker(dspy.Module):
    """
    Phase 6: Replaces per-event poignancy LLM calls.
    Receives a pre-filtered list of candidate MemoryEntry objects and
    returns them ranked by contextual relevance in a single LLM call.
    This module is DSPy-optimizable via BootstrapFewShot / MIPRO.
    """

    def __init__(self):
        super().__init__()
        # thinking_lm: ranking memories requires nuanced multi-step reasoning.
        self.predict = dspy.Predict(ContextualRerankSignature)

    def forward(
        self,
        agent_name: str,
        current_situation: str,
        current_plan: str,
        candidates: list[Any],
        top_n: int = 10,
    ) -> list[Any]:
        """
        candidates: list of objects that have at least a `.content` or `.description` str
                    attribute and optionally a `.depth` int attribute.
        Returns the same objects re-ordered by contextual relevance.
        """
        if not candidates:
            return []

        # Build numbered list with level annotation
        lines = []
        for i, m in enumerate(candidates):
            text = getattr(m, "content", None) or getattr(m, "description", str(m))
            level = getattr(m, "depth", getattr(m, "level", 0))
            lines.append(f"{i}. [L{level}] {text}")
        candidate_str = "\n".join(lines)

        try:
            with dspy.context(lm=thinking_lm) if thinking_lm else dspy.context():
                result = self.predict(
                agent_name=agent_name,
                current_situation=current_situation or "idle",
                current_plan=current_plan or "no plan",
                candidate_memories=candidate_str,
            )
            indices = result.ranked_indices
            # Validate and deduplicate indices
            seen = set()
            valid = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    seen.add(idx)
                    valid.append(idx)
            # Append any missing indices at the end (unranked remainder)
            for idx in range(len(candidates)):
                if idx not in seen:
                    valid.append(idx)
            return [candidates[i] for i in valid[:top_n]]
        except Exception:
            # Fallback: return candidates as-is
            return candidates[:top_n]
