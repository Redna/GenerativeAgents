import dspy
from pydantic import BaseModel, Field


class PlanOutline(BaseModel):
    hour: int = Field(description="The hour of the day (1-24).", ge=1, le=24)
    description: str = Field(description="Brief description of activity.")


class FirstDailyPlanSignature(dspy.Signature):
    """
    Create a broad, high-level plan for the day.
    """

    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    today: str = dspy.InputField(desc="Today's date.")
    wake_up_hour: str = dspy.InputField(desc="Wake up time.")

    plan_in_broad_strokes: list[PlanOutline] = dspy.OutputField(
        desc="List of broad daily activities."
    )


def create_daily_plan(
    name: str, identity: str, today: str, wake_up_hour: str
) -> list[dict[str, str]]:
    try:
        predict = dspy.ChainOfThought(FirstDailyPlanSignature)
        response = predict(
            name=name, identity=identity, today=today, wake_up_hour=wake_up_hour
        )

        converted_plan = {}
        if hasattr(response, "plan_in_broad_strokes"):
            for activity in response.plan_in_broad_strokes:
                if isinstance(activity, dict):
                    h = activity.get("hour", 12)
                    d = activity.get("description", "")
                else:
                    h = activity.hour
                    d = activity.description

                # Convert to 12h format as in original code
                orig_h = h
                h = h % 12
                meridien = (
                    "AM" if orig_h < 12 and orig_h != 24 else "PM"
                )  # Basic logic, improving original
                if h == 0:
                    h = 12

                time_str = f"{str(h).zfill(2)}:00 {meridien}"
                converted_plan[time_str] = d

        return converted_plan
    except Exception as e:
        print(f"Error in create_daily_plan: {e}")
        return {}


if __name__ == "__main__":
    if not dspy.settings.lm:
        dspy.settings.configure(
            lm=dspy.DummyLM(
                [
                    {
                        "plan_in_broad_strokes": [
                            PlanOutline(hour=8, description="Wake up")
                        ]
                    }
                ]
            )
        )
    # No main execution in original file really, just defs? actually the 
    # original file has imports but no example execution in __main__
    pass
