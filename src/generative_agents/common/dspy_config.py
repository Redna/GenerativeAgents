"""
common/dspy_config.py
Configures two DSPy LM instances for Qwen3-8B running on vLLM:
  - thinking_lm : enable_thinking=True  — for complex reasoning (planning, reflection, reranking)
  - fast_lm     : enable_thinking=False — for structured extraction (actor, triples, summaries)

Why dspy.Predict instead of dspy.ChainOfThought?
  ChainOfThought adds an explicit rationale OUTPUT FIELD to the prompt, making the model
  write its reasoning visibly. Qwen3-8B already reasons inside <think> tokens managed by
  vLLM. Using Predict avoids double-reasoning and keeps outputs clean.
"""
import dspy

# These are module-level singletons — import and use directly.
thinking_lm: dspy.LM | None = None
fast_lm: dspy.LM | None = None


def configure_dspy():
    """
    Initialises both LM configs and sets the default to fast_lm.
    Modules that need deep reasoning call `with dspy.context(lm=thinking_lm):`.
    """
    global thinking_lm, fast_lm

    _base = dict(
        model="openai/Qwen/Qwen3-8B",
        api_base="http://pop-os:8000/v1",
        api_key="EMPTY",
    )

    thinking_lm = dspy.LM(
        **_base,
        # Qwen3 (non-VL) supports per-request enable_thinking
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    fast_lm = dspy.LM(
        **_base,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # Default: fast_lm (thinking disabled globally; modules opt in via context)
    dspy.configure(lm=fast_lm)
    print("DSPy configured: fast_lm=Qwen3-8B(thinking=False), thinking_lm=Qwen3-8B(thinking=True)")
