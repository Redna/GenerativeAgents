import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from generative_agents.agents.agent import Agent
from generative_agents.common.percept import Percept
from generative_agents.simulation.time import SimulationTime
from generative_agents.simulation.maze import Maze, Tile, Level
from generative_agents.common.events import PerceivedEvent, EventType
import dspy

class DummyLM(dspy.LM):
    def __init__(self, responses):
        super().__init__("dummy")
        self.responses = responses
        self.count = 0

    def __call__(self, prompt=None, **kwargs):
        # Handle older DSPy versions or different call signatures
        if prompt is None:
            prompt = kwargs.get("prompt", "")
        return self.basic_request(prompt, **kwargs)

    def basic_request(self, prompt, **kwargs):
        print(f"DummyLM Prompt used: {prompt[:200]}...")
        
        prompt_lower = prompt.lower()
        if "insight" in prompt_lower or "evidence" in prompt_lower:
             # Mock response for evidence_and_insights
             return ['{"Klaus is diligent": ["reading book", "studying"]}']
        
        if "identity" in prompt_lower or "personality" in prompt_lower:
            # Mock response for formulate_identity in CoT text format
            return ["Reasoning: Based on the context provided.\nIdentity: Klaus is a dedicated student who loves to read and study."]
            
        if "rate" in prompt_lower and "poignance" in prompt_lower:
            return ['{"rating": 8}']
            
        if "subject" in prompt_lower and "predicate" in prompt_lower:
             return ['{"subject": "Klaus", "predicate": "is", "object": "testing"}']
        
        # Return a Super JSON that satisfies all signatures
        return ['''
        {
            "reasoning": "Standard routine", 
            "wake_up_hour": "7:00 am",
            "plan_in_broad_strokes": "wake up, eat breakfast, read book",
            "schedule": [["wake up", 60], ["eat breakfast", 60], ["read book", 120]],
             "focal_points": ["studying", "eating", "socializing"],
             "identity": "Klaus is a dedicated student (Fallback)",
             "rating": 5,
             "subject": "Agent",
             "predicate": "act",
             "object": "object",
             "emoji": "🧪"
        }
        ''']

def main():
    print("Initializing Neural Agent Verification...")
    
    # Configure Dummy LM
    lm = DummyLM([]) 
    dspy.configure(lm=lm)
    time = SimulationTime(datetime.datetime(2023, 2, 13, 8, 0, 0))
    maze = Maze() 
    
    # --- Perception Module Verification ---
    print("\n--- Verifying Perception Module ---")
    from generative_agents.intelligence.modules.perception import PoignanceRater, EventParser
    
    rater = PoignanceRater()
    score = rater("Klaus", "Student", "test", "Klaus is testing")
    print(f"Poignance Score: {score}")
    
    parser = EventParser()
    triple = parser.get_triple("Klaus", "Klaus is testing the brain")
    print(f"Triple: {triple}")
    
    # --- Dialogue Module Verification ---
    print("\n--- Verifying Dialogue Module ---")
    from generative_agents.intelligence.modules.dialogue import TalkDecider, DialogueGenerator
    
    talk = TalkDecider()
    decision = talk("context", "12:00", "Klaus", "Maria", "none", "idle", "idle")
    print(f"Decide to talk: {decision}")
    
    gen = DialogueGenerator()
    utt, end = gen("Klaus", "Student", "mem", "ctx", "school", "idle", "Maria", "idle", "Hi")
    print(f"Utterance: {utt} (End: {end})")
    
    # --- Reflection Module Verification ---
    print("\n--- Verifying Reflection Module ---")
    from generative_agents.intelligence.modules.reflection import ReflectionPointGenerator, InsightGenerator
    
    ref_gen = ReflectionPointGenerator()
    questions = ref_gen("Klaus is studying hard. Klaus is tired.", 2)
    print(f"Questions: {questions}")
    
    ins_gen = InsightGenerator()
    insights = ins_gen(["Klaus reads a lot.", "Klaus visits library."], 1)
    print(f"Insights: {insights}")
    
    # --- Spatial Module Verification ---
    print("\n--- Verifying Spatial Module ---")
    from generative_agents.intelligence.modules.spatial import SectorSelector, ArenaSelector, ObjectSelector
    
    sector_sel = SectorSelector()
    sector = sector_sel("Klaus", "Home", "Bed, Bath", "School", "Class, Lab", "Park", "Going to class")
    print(f"Sector: {sector}")
    
    arena_sel = ArenaSelector()
    arena = arena_sel("Klaus", "Class", "School", "School", "Library, Gym", "Study")
    print(f"Arena: {arena}")
    
    obj_sel = ObjectSelector()
    obj = obj_sel("Writing code", "Computer, Desk, Chair")
    print(f"Object: {obj}")
    # ------------------------------------- 
    
    # 2. Setup Agent
    print("Creating Agent...")
    # x, y, world, sector, arena, game_object, spawning_location, collision, events
    home_tile = Tile(127, 46, "dorm", "room", "bed", "bed-1", "spawning_loc", False, set())
    
    agent = Agent(
        name="Klaus_Mueller",
        age=20,
        description="A student at Oak Hill College.",
        innate_traits=["friendly", "studious"],
        time=time,
        location="Oak Hill College",
        emoji="🧑‍🎓",
        activity="sleeping",
        tile=home_tile
    )
    
    # 3. Create Percept
    print("Creating Percept...")
    ignored_event = PerceivedEvent(
        event_type=EventType.EVENT,
        depth=1,
        description="The sun is rising",
        created=time.time,
        expiration=time.time + datetime.timedelta(hours=1),
        subject="Sun",
        predicate="is",
        object_="rising"
    )
    
    percept = PerceivedEvent(
        event_type=EventType.EVENT,
        depth=1, 
        description="Test Percept",
        created=time.time,
        expiration=None,
        subject="Agent",
        predicate="is",
        object_="testing"
    )

    percept = Percept(
        events=[ignored_event, percept],
        nearby_tiles=[]
    )
    
    # 4. Run Step 1 (Brain Forward Pass)
    print("Running Agent.run_step()...")
    try:
        next_tile = agent.run_step(percept, maze, {}, time)
        print(f"Step 1 Complete. Next Tile: {next_tile}")
        print(f"Current Action: {agent.working_memory.action.address if agent.working_memory.action else 'None'}")
        print(f"Daily Plan: {agent.working_memory.daily_requirements}")
        print(f"Schedule Items: {len(agent.working_memory.daily_schedule) if agent.working_memory.daily_schedule else 0}")
    except Exception as e:
        print(f"Step 1 Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Run Step 2 (Wait/Action)
    print("\n--- Running Step 2 ---")
    time.time += datetime.timedelta(minutes=10) # Advance time
    
    # FORCE SYSTEM 2: Reflection
    print("Forcing Reflection Trigger...")
    agent.working_memory.reflection_trigger_counter = 0 # Force trigger
    
    try:
        next_tile = agent.run_step(percept, maze, {}, time)
        print(f"Step 2 Complete. Next Tile: {next_tile}")
        print(f"Identity Description: {agent.working_memory.identity_description}")
        if "dedicated student" in agent.working_memory.identity_description:
            print("SUCCESS: Identity updated by System 2!")
        else:
            print("WARNING: Identity NOT updated.")
            
    except Exception as e:
        print(f"Step 2 Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
