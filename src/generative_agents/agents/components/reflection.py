import datetime
from typing import TYPE_CHECKING

from generative_agents.common.events import EventType, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.intelligence.action_event_tripple import action_event_triple
from generative_agents.intelligence.evidence_and_insights import evidence_and_insights
from generative_agents.intelligence.memo_on_conversation import memo_on_conversation
from generative_agents.intelligence.planning_on_conversation import (
    planning_on_conversation,
)
from generative_agents.intelligence.poignance import rate_poignance
from generative_agents.intelligence.reflection_points import reflection_points
from generative_agents.persistence.database import ConversationFilling

if TYPE_CHECKING:
    from generative_agents.agents.agent import Agent


class Reflection:
    def __init__(self, agent: "Agent"):
        self.agent = agent
        # 1. Reflect on conversation
        if self._should_reflect_on_conversation():
            self._reflect_on_conversation()

        # 2. General Reflection
        if self.agent.scratch.should_reflect():
            self._run_reflect()

    def _should_reflect_on_conversation(self) -> bool:
        last_conversation = self.agent.associative_memory.last_conversation_with(
            self.agent.scratch.chatting_with
        )
        return (
            last_conversation
            and last_conversation.filling
            and last_conversation.filling[-1].end
        )

    def _reflect_on_conversation(self):
        last_conversation = self.agent.associative_memory.last_conversation_with(
            self.agent.scratch.chatting_with
        )

        if (
            last_conversation
            and last_conversation.filling
            and last_conversation.filling[-1].end
        ):
            evidence = [last_conversation.id]

            planning_thought = self._generate_planning_thought_on_conversation(
                last_conversation.filling
            )
            log_agent(
                self.agent.name, f"planning thought is {planning_thought}", "DEBUG"
            )
            planning_thought = (
                f"For {self.agent.scratch.name}'s planning: {planning_thought}"
            )
            self._add_reflection_thought(planning_thought, evidence)
            log_agent(self.agent.name, "added reflection thought", "DEBUG")

            memo_thought = self._generate_memo_on_conversation(
                last_conversation.filling
            )
            memo_thought = f"{self.agent.name} {memo_thought}"
            log_agent(self.agent.name, f"memo thought is {memo_thought}", "DEBUG")
            self._add_reflection_thought(memo_thought, evidence)

    def _run_reflect(self):
        """
        Run the actual reflection. We generate the focal points, retrieve any
        relevant nodes, and generate thoughts and insights.

        INPUT:
            persona: Current Persona object
        Output:
            None
        """
        # Reflection requires certain focal points. Generate that first.
        focal_points = self._generate_reflection_points(3)
        log_agent(self.agent.name, f"generated {focal_points} focal points", "DEBUG")
        # Retrieve the relevant Nodes object for each of the focal points.
        # <retrieved> has keys of focal points, and values of the associated Nodes.
        retrieved = self.agent.associative_memory.retrieve_relevant_entries(
            focal_points
        )

        log_agent(
            self.agent.name, f"retrieved {len(retrieved)} relevant nodes", "DEBUG"
        )

        # For each of the focal points, generate thoughts and save it in the
        # agent's memory.
        for nodes in retrieved:
            thoughts = self._generate_insights_and_evidence(nodes, 5)
            for thought, evidence in thoughts.items():
                self._add_reflection_thought(thought, evidence)

        self.agent.scratch.reset_reflection_counter()

    def _generate_reflection_points(self, num_points: int):
        memories = self.agent.associative_memory.get_most_recent_memories(num_points)
        return reflection_points(memory=memories, count=num_points)

    def _generate_insights_and_evidence(
        self, memories: list[PerceivedEvent], num_insights: int
    ):
        """
        Generate insights and evidence for the given memories.
        INPUT:
            memories: A list of <PerceivedEvent> instances that are the memories
                that we want to generate insights for.
            num_insights: The number of insights that we want to generate.
        OUTPUT:
            insights: A dictionary that contains the generated insights.
                insights[insight] = evidence
        """

        # TODO parse the output properly
        statements = "\n".join(
            [
                f"{str(count)}. {node.embedding_key}"
                for count, node in enumerate(memories, 1)
            ]
        )

        return evidence_and_insights(
            statements=statements, number_of_insights=num_insights
        )

    def _add_reflection_thought(self, thought: str, evidence: list[str]):
        created = self.agent.scratch.time
        expiration = created + datetime.timedelta(days=30)
        s, p, o = action_event_triple(self.agent.name, thought)

        thought_poignancy = self._rate_perception_poignancy(EventType.THOUGHT, thought)

        perceived_event = PerceivedEvent(
            subject=s,
            predicate=p,
            object_=o,
            description=thought,
            event_type=EventType.THOUGHT,
            poignancy=thought_poignancy,
            filling=evidence,
            expiration=expiration,
            created=created,
        )
        self.agent.associative_memory.add(perceived_event)

    def _rate_perception_poignancy(
        self, event_type: EventType, description: str
    ) -> float:
        if "idle" in description:
            return 0.1

        score = rate_poignance(
            self.agent.name, self.agent.scratch.identity, event_type.value, description
        )

        return int(score) / 10

    def _generate_memo_on_conversation(self, utterances: list[ConversationFilling]):
        return memo_on_conversation(
            agent=self.agent.name,
            conversation=self.__utterances_to_conversation(utterances),
        )

    def _generate_planning_thought_on_conversation(
        self, utterances: list[ConversationFilling]
    ):
        return planning_on_conversation(
            agent=self.agent.name,
            conversation=self.__utterances_to_conversation(utterances),
        )

    @staticmethod
    def __utterances_to_conversation(utterances: list[ConversationFilling]):
        return "\n".join([f"{i.name}: {i.utterance}" for i in utterances])
