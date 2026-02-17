import dspy

class ActionLocationArenaSignature(dspy.Signature):
    """
    Identify the next area for a character within a sector.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    current_area: str = dspy.InputField(desc="Current area of the agent.")
    current_sector: str = dspy.InputField(desc="Current sector.")
    sector: str = dspy.InputField(desc="Target sector.")
    available_arenas: str = dspy.InputField(desc="Comma-separated list of available arenas in the target sector.")
    action_description: str = dspy.InputField(desc="Description of the action.")
    
    reasoning: str = dspy.OutputField(desc="Reasoning for the area selection.")
    next_area: str = dspy.OutputField(desc="The selected area from available_arenas.")

def action_area_locations(name: str, current_area: str, current_sector: str, sector: str, sector_arenas: str, action_description: str) -> str:
    try:
        predict = dspy.ChainOfThought(ActionLocationArenaSignature)
        response = predict(
            name=name,
            current_area=current_area,
            current_sector=current_sector,
            sector=sector,
            available_arenas=sector_arenas,
            action_description=action_description
        )
        
        # Validation: check if response is in allowed arenas
        # Relaxed matching could be added, but exact match for now
        allowed = [a.strip() for a in sector_arenas.split(",")]
        cleaned_response = response.next_area.strip()
        
        # Basic fuzzy matching or fallback
        for a in allowed:
            if a.lower() == cleaned_response.lower():
                return a
        if allowed: return allowed[0] # Fallback to first
        return ""
        
    except Exception as e:
        print(f"Error in action_area_locations: {e}")
        return sector_arenas.split(",")[0] if sector_arenas else ""

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "next_area": "bedroom"
         }]))
    print(action_area_locations(name="John Doe", 
                                current_area="common room", 
                                current_sector="John Doe's apartment", 
                                sector="Hobbs Cafe", 
                                sector_arenas="kitchen, bedroom, bathroom", 
                                action_description="Putting on trousers"))