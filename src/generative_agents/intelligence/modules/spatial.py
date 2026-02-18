import dspy

# --- Signatures ---

class ActionLocationSectorSignature(dspy.Signature):
    """Choose an appropriate sector from the available options for a given activity."""
    agent_name: str = dspy.InputField(desc="Name of the agent.")
    agent_home: str = dspy.InputField(desc="Agent's home sector.")
    agent_home_arenas: str = dspy.InputField(desc="Arenas in home sector.")
    agent_current_sector: str = dspy.InputField(desc="Current sector.")
    agent_current_sector_arenas: str = dspy.InputField(desc="Arenas in current sector.")
    available_sectors_nearby: str = dspy.InputField(desc="Nearby sectors.")
    action_description: str = dspy.InputField(desc="Description of the action.")
    
    reasoning: str = dspy.OutputField(desc="Reasoning for the sector selection.")
    next_sector: str = dspy.OutputField(desc="The selected sector.")

class ActionLocationArenaSignature(dspy.Signature):
    """Identify the next area for a character within a sector."""
    name: str = dspy.InputField(desc="Name of the agent.")
    current_area: str = dspy.InputField(desc="Current area of the agent.")
    current_sector: str = dspy.InputField(desc="Current sector.")
    sector: str = dspy.InputField(desc="Target sector.")
    available_arenas: str = dspy.InputField(desc="Comma-separated list of available arenas in the target sector.")
    action_description: str = dspy.InputField(desc="Description of the action.")

    reasoning: str = dspy.OutputField(desc="Reasoning for the area selection.")
    next_area: str = dspy.OutputField(desc="The selected area from available_arenas.")

class ActionLocationGameObjectSignature(dspy.Signature):
    """Identify the most relevant object for an action from the available objects."""
    action_description: str = dspy.InputField(desc="Current activity description.")
    available_objects: str = dspy.InputField(desc="Comma-separated list of available objects.")
    next_object: str = dspy.OutputField(desc="The most relevant object selected from the list.")

# --- Modules ---

class SectorSelector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ActionLocationSectorSignature)

    def forward(self, agent_name: str, agent_home: str, agent_home_arenas: str, 
                agent_current_sector: str, agent_current_sector_arenas: str, 
                available_sectors_nearby: str, curr_action_description: str) -> str:
        
        possible_sectors = [
            s.strip()
            for s in ",".join([agent_home, agent_current_sector, available_sectors_nearby])
            .replace(", ", ",")
            .split(",")
            if s
        ]

        try:
            response = self.predict(
                agent_name=agent_name, agent_home=agent_home, agent_home_arenas=agent_home_arenas,
                agent_current_sector=agent_current_sector, agent_current_sector_arenas=agent_current_sector_arenas,
                available_sectors_nearby=available_sectors_nearby, action_description=curr_action_description
            )
            
            cleaned_response = response.next_sector.strip()
            for s in possible_sectors:
                if s.lower() == cleaned_response.lower():
                    return s
            return possible_sectors[0] if possible_sectors else ""
        except Exception:
            return possible_sectors[0] if possible_sectors else ""

class ArenaSelector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ActionLocationArenaSignature)

    def forward(self, name: str, current_area: str, current_sector: str, 
                sector: str, sector_arenas: str, action_description: str) -> str:
        try:
            response = self.predict(
                name=name, current_area=current_area, current_sector=current_sector,
                sector=sector, available_arenas=sector_arenas, action_description=action_description
            )
            
            allowed = [a.strip() for a in sector_arenas.split(",")]
            cleaned_response = response.next_area.strip()
            for a in allowed:
                if a.lower() == cleaned_response.lower():
                    return a
            return allowed[0] if allowed else ""
        except Exception:
            return sector_arenas.split(",")[0] if sector_arenas else ""

class ObjectSelector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ActionLocationGameObjectSignature)

    def forward(self, action_description: str, available_objects: str) -> str:
        try:
            response = self.predict(
                action_description=action_description, available_objects=available_objects
            )
            
            allowed = [o.strip() for o in available_objects.split(",")]
            cleaned = response.next_object.strip()
            for o in allowed:
                if o.lower() == cleaned.lower():
                    return o
            return allowed[0] if allowed else ""
        except Exception:
            return available_objects.split(",")[0] if available_objects else ""
