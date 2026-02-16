import dspy
from pydantic import BaseModel, Field

# --- Signatures ---

class RememberSignature(dspy.Signature):
    """
    Identify the most important things the agent should remember to create their daily plan.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    statements: str = dspy.InputField(desc="Relevant statements/memories.")
    today: str = dspy.InputField(desc="Current date.")
    
    things_i_should_remember: str = dspy.OutputField(desc="Brief overview of things to remember for planning.")

class FeelingsSignature(dspy.Signature):
    """
    Summarize the agent's feelings about their days up to now.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    statements: str = dspy.InputField(desc="Relevant statements/memories.")
    
    feelings: str = dspy.OutputField(desc="Brief summary of feelings.")

class ReflectionsSignature(dspy.Signature):
    """
    Reflect on yesterday to determine status for today.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    yesterday: str = dspy.InputField(desc="Yesterday's date.")
    today: str = dspy.InputField(desc="Today's date.")
    current_activity: str = dspy.InputField(desc="Status from yesterday.")
    thought_note: str = dspy.InputField(desc="Thoughts at end of yesterday.")
    plan_note: str = dspy.InputField(desc="Plan note from yesterday.")
    
    reflection_of_yestereday: str = dspy.OutputField(desc="Brief reflection of yesterday's thoughts in third-person.")

class PlanOutline(BaseModel):
    hour: int = Field(description="The hour of the day. (0-23)", ge=0, le=23)
    description: str = Field(description="A brief description of the activity planned for the hour.")

class DailyPlanSignature(dspy.Signature):
    """
    Create a broad-strokes daily plan based on feelings and identity.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    today: str = dspy.InputField(desc="Today's date.")
    feelings_for_today: str = dspy.InputField(desc="Reflected feelings/status for today.")
    
    plan_in_broad_strokes: list[PlanOutline] = dspy.OutputField(desc="List of planned activities with hours.")

# --- Functions ---

def find_things_to_remember(name: str, identity: str, statements: str, today: str) -> str:
    try:
        predict = dspy.Predict(RememberSignature)
        response = predict(name=name, identity=identity, statements=statements, today=today)
        return response.things_i_should_remember
    except Exception as e:
        print(f"Error in find_things_to_remember: {e}")
        return ""

def find_feelings(name: str, identity: str, statements: str) -> str:
    try:
        predict = dspy.Predict(FeelingsSignature)
        response = predict(name=name, identity=identity, statements=statements)
        return response.feelings
    except Exception as e:
        print(f"Error in find_feelings: {e}")
        return ""

def define_current_status(name: str, yesterday: str, today: str, current_activity: str, thought_note: str, plan_note: str, identity: str) -> str:
    try:
        predict = dspy.Predict(ReflectionsSignature)
        response = predict(
            name=name, yesterday=yesterday, today=today, current_activity=current_activity,
            thought_note=thought_note, plan_note=plan_note, identity=identity
        )
        return response.reflection_of_yestereday
    except Exception as e:
        print(f"Error in define_current_status: {e}")
        return ""

def create_daily_plan(name: str, identity: str, today: str, feelings_for_today: str) -> dict[str, str]:
    try:
        # Using ChainOfThought instead of TypedPredictor as TypedPredictor is missing
        predict = dspy.ChainOfThought(DailyPlanSignature)
        response = predict(name=name, identity=identity, today=today, feelings_for_today=feelings_for_today)
        
        converted_plan = {}
        # response.plan_in_broad_strokes might be returned as part of the prediction object
        if hasattr(response, 'plan_in_broad_strokes'):
             for activity in response.plan_in_broad_strokes:
                # Ensure activity is an object with hour and description
                # If dspy returns dicts instead of Pydantic objects, handle that
                if isinstance(activity, dict):
                    hour_val = activity.get('hour', 0)
                    desc_val = activity.get('description', '')
                else:
                    hour_val = activity.hour
                    desc_val = activity.description
                
                hour_int = hour_val % 24 # Ensure 0-23
                
                display_hour = hour_int % 12
                if display_hour == 0: display_hour = 12
                
                meridien = "AM" if hour_int < 12 else "PM"
                time_str = f"{str(display_hour).zfill(2)}:00 {meridien}"
                
                # Simple conflict resolution: overwrite
                converted_plan[time_str] = desc_val
            
        return converted_plan
    except Exception as e:
        print(f"Error in create_daily_plan: {e}")
        return {}

def create_daily_plan_and_status(name: str, identity: str, today: str, yesterday: str, statements: str, current_activity: str) -> tuple[str, str]:
    remember = find_things_to_remember(name, identity, statements, today)
    feelings = find_feelings(name, identity, statements)

    feelings_for_today = define_current_status(
        name, yesterday, today, current_activity, feelings, remember, identity)

    daily_plan = create_daily_plan(name, identity, today, feelings_for_today)

    plan = " ".join([f"{i+1}) {activity} at {time}" for i,
                    (time, activity) in enumerate(daily_plan.items())])

    return plan, feelings_for_today

if __name__ == "__main__":
    # Setup dummy for testing
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.DummyLM([{
            "things_i_should_remember": "Nothing specific.", 
            "feelings": "Good.",
            "reflection_of_yestereday": "He felt good.",
            "plan_in_broad_strokes": [PlanOutline(hour=9, description="Wake up")]
        }]))

    print(create_daily_plan_and_status(name="John Doe",
                                       identity="John Doe is a 30 year old software developer...",
                                       today="2023-08-01",
                                       yesterday="2023-07-31",
                                       statements="Yesterday was productive..."))
