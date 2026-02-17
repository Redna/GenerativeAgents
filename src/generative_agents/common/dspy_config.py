import dspy


def configure_dspy():
    """
    Configures the DSPy language model.
    Uses Ollama with 'qwen3:4b' (or similar) via dspy.LM as requested.
    """
    # Using the pattern provided by the user for dspy >= 3.1.3
    lm = dspy.LM(
        "ollama_chat/qwen2.5:4b", api_base="http://localhost:11434", api_key=""
    )
    dspy.configure(lm=lm)
    print("DSPy configured with Ollama (qwen2.5:4b)")
