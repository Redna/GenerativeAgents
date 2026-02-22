"""
intelligence/modules/perception.py
Phase 6: Perception utilities.
- PoignanceRater replaced with heuristic_poignance() — zero LLM calls.
- EmojiMapper replaced with lookup table — zero LLM calls.
- EventParser kept (triple extraction, 1 LLM call, used in ActorLayer dispatch).
- ReactionDecider removed (replaced by ReActActor speak_to/wait tools).
- Contextualizer removed (no callers).
"""
import re
import dspy
from typing import Tuple

from generative_agents.common.events import EventType


# ---------------------------------------------------------------------------
# Heuristic Poignance Scoring (0 LLM calls)
# ---------------------------------------------------------------------------

# Base scores by event type
_TYPE_BASE: dict[str, float] = {
    EventType.CHAT.value:        0.7,
    "chat":                      0.7,
    EventType.EVENT.value:       0.4,
    "event":                     0.4,
    EventType.PLAN.value:        0.3,
    "plan":                      0.3,
    EventType.OBSERVATION.value: 0.3,
    "observation":               0.3,
}

# Keywords that modify the base score
_HIGH_SIGNAL = [
    "died", "death", "killed", "accident", "emergency", "fired", "married",
    "promoted", "argument", "fight", "crisis", "arrest", "diagnosed",
    "invited", "meet", "plan", "canceled", "changed", "cancelled"
]
_LOW_SIGNAL = [
    "sleeping", "idle", "waiting", "eating", "walking",
]


def heuristic_poignance(event_type: str, description: str) -> float:
    """
    Returns a contextual importance score in [0.0, 1.0] without any LLM call.
    Used by SensoryProcessingLayer to score incoming percepts before retrieval.
    """
    base = _TYPE_BASE.get(event_type.lower(), 0.4)
    text = description.lower()

    boost = sum(0.15 for kw in _HIGH_SIGNAL if kw in text)
    penalty = sum(0.06 for kw in _LOW_SIGNAL if kw in text)

    score = base + boost - penalty
    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Emoji lookup table (0 LLM calls)
# ---------------------------------------------------------------------------

_EMOJI_MAP: list[tuple[list[str], str]] = [
    (["sleep", "nap", "rest", "bed"],                         "😴"),
    (["eat", "breakfast", "lunch", "dinner", "meal", "food"], "🍽️"),
    (["work", "study", "read", "research", "write", "code"],  "💼"),
    (["walk", "run", "jog", "exercise", "gym"],               "🚶"),
    (["chat", "talk", "convers", "discuss", "meet"],          "💬"),
    (["shop", "buy", "market", "store"],                      "🛍️"),
    (["cook", "bak", "prepar"],                               "🍳"),
    (["clean", "tidy", "wash"],                               "🧹"),
    (["play", "game", "fun", "entertain"],                    "🎮"),
    (["meditat", "relax", "yoga"],                            "🧘"),
    (["watch", "tv", "movie", "film"],                        "📺"),
    (["draw", "paint", "art", "creat"],                       "🎨"),
    (["music", "sing", "danc"],                               "🎵"),
    (["call", "phone"],                                       "📞"),
    (["travel", "commute", "drive"],                          "🚗"),
]


def heuristic_emoji(action_description: str) -> str:
    """Returns a relevant emoji for an action description without any LLM call."""
    text = action_description.lower()
    for keywords, emoji in _EMOJI_MAP:
        if any(kw in text for kw in keywords):
            return emoji
    return "⚡"


