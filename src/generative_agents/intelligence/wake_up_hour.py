import dspy


class WakeUpHourSignature(dspy.Signature):
    """
    Estimate the wake up hour of an agent based on their identity and lifestyle.
    """

    agent_name: str = dspy.InputField(desc="Name of the agent.")
    agent_identity: str = dspy.InputField(desc="Identity and backstory of the agent.")
    agent_lifestyle: str = dspy.InputField(desc="Lifestyle details of the agent.")

    rationale: str = dspy.OutputField(
        desc="Maximum two sentences reason for the wake up hour."
    )
    wake_up_hour: int = dspy.OutputField(desc="Wake up hour in 24-hour format (0-23).")


def estimate_wake_up_hour(
    agent_name: str, agent_identity: str, agent_lifestyle: str
) -> str:
    try:
        predict = dspy.ChainOfThought(WakeUpHourSignature)
        response = predict(
            agent_name=agent_name,
            agent_identity=agent_identity,
            agent_lifestyle=agent_lifestyle,
        )

        hour = response.wake_up_hour
        # Helper to format output as expected by original code
        formatted_hour = str(hour).zfill(2) + ":00 " + ("AM" if hour < 12 else "PM")
        return formatted_hour
    except Exception as e:
        print(f"Error in estimate_wake_up_hour: {e}")
        return "07:00 AM"  # Default safe value


if __name__ == "__main__":
    print(
        estimate_wake_up_hour(
            "Emily Clark",
            "Emily Clark is a 25 year old freelance graphic designer. "
            "She adores creating vibrant and unique designs. "
            "Emily is an avid traveler, seeking inspiration from different cultures around the world. "
            "She prefers working late nights to capture the essence of her travels in her designs.",
            "Emily finds her creativity peaks in the quiet of the night.",
        )
    )

    print(
        estimate_wake_up_hour(
            "Michael Thompson",
            "Michael Thompson is a 40 year old professional photographer. "
            "He specializes in wildlife photography and spends months in remote locations "
            "capturing the beauty of nature. Michael is an adventure seeker and enjoys hiking and camping.",
            "Michael is accustomed to waking up at dawn to catch the perfect light for his photographs.",
        )
    )

    print(
        estimate_wake_up_hour(
            "Linda Wu",
            "Linda Wu is a 35 year old entrepreneur running her own start-up. "
            "She is dedicated to creating eco-friendly products. "
            "Linda is passionate about sustainability and environmental conservation. "
            "She practices yoga daily to stay focused and energized.",
            "Linda is an early riser, finding the morning the best time to plan her day and meditate.",
        )
    )
