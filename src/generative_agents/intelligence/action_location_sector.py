import dspy


class ActionLocationSectorSignature(dspy.Signature):
    """
    Choose an appropriate sector from the available options for a given activity.
    """

    agent_name: str = dspy.InputField(desc="Name of the agent.")
    agent_home: str = dspy.InputField(desc="Agent's home sector.")
    agent_home_arenas: str = dspy.InputField(desc="Arenas in home sector.")
    agent_current_sector: str = dspy.InputField(desc="Current sector.")
    agent_current_sector_arenas: str = dspy.InputField(desc="Arenas in current sector.")
    available_sectors_nearby: str = dspy.InputField(desc="Nearby sectors.")
    action_description: str = dspy.InputField(desc="Description of the action.")

    reasoning: str = dspy.OutputField(desc="Reasoning for the sector selection.")
    next_sector: str = dspy.OutputField(desc="The selected sector.")


def action_sector_locations(
    agent_name: str,
    agent_home: str,
    agent_home_arenas: str,
    agent_current_sector: str,
    agent_current_sector_arenas: str,
    available_sectors_nearby: str,
    curr_action_description: str,
) -> str:
    possible_sectors = [
        s.strip()
        for s in ",".join([agent_home, agent_current_sector, available_sectors_nearby])
        .replace(", ", ",")
        .split(",")
        if s
    ]

    try:
        predict = dspy.ChainOfThought(ActionLocationSectorSignature)
        response = predict(
            agent_name=agent_name,
            agent_home=agent_home,
            agent_home_arenas=agent_home_arenas,
            agent_current_sector=agent_current_sector,
            agent_current_sector_arenas=agent_current_sector_arenas,
            available_sectors_nearby=available_sectors_nearby,
            action_description=curr_action_description,
        )

        cleaned_response = response.next_sector.strip()
        for s in possible_sectors:
            if s.lower() == cleaned_response.lower():
                return s

        if possible_sectors:
            return possible_sectors[0]
        return ""
    except Exception as e:
        print(f"Error in action_sector_locations: {e}")
        return possible_sectors[0] if possible_sectors else ""


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.DummyLM([{"next_sector": "Hobbs Cafe"}]))
    print(
        action_sector_locations(
            agent_name="Jimmy Foe",
            agent_home="Jimmy Foe's apartment",
            agent_home_arenas="living room, bathroom",
            agent_current_sector="Hobbs Cafe",
            agent_current_sector_arenas="cafe, restroom",
            available_sectors_nearby="Supermarket, Library, Lyn's family room",
            curr_action_description="drinking a cafe",
        )
    )
