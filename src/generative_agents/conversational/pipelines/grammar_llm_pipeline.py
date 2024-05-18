
from haystack import Pipeline, component
from haystack.components.builders import DynamicPromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator


from llama_cpp import LlamaGrammar
from pydantic import BaseModel


def pydantic_to_grammar(model: BaseModel) -> LlamaGrammar:
    schema = model.model_json_schema()

    return LlamaGrammar.from_json_schema(schema)


@component
class GrammarGenerator:
    @component.output_types(generation_kwargs=dict[str, any])
    def run(self, model: BaseModel):
        schema = model.model_json_schema()
        return {"generation_kwargs": {"grammar": LlamaGrammar.from_json_schema(schema)}}

@component
class LLMOutputParser:
    @component.output_types(model=BaseModel)
    def run(self, model: BaseModel, replies: list[str]):
        return {"model": model.model_validate_json(replies)}

class _GrammarPipeline:
    def __init__(self):
        self.pipe = Pipeline()

        generator = LlamaCppGenerator(model="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)
        self.pipe.add_component("prompt", instance=DynamicPromptBuilder())

        self.pipe.add_component("grammar_generator", GrammarGenerator())
        self.pipe.add_component("llm", generator)
        self.pipe.add_component("output_parser", LLMOutputParser())

        self.pipe.connect("prompt.prompt", "llm.prompt")
        self.pipe.connect("grammar_generator", "llm.generation_kwargs")
        self.pipe.connect("llm.replies", "output_parser.replies")
    
    def run(self, model: BaseModel, prompt_template: str, template_variables: dict[str, any]):
        return self.pipe.run(data={
            "prompt": {
                "prompt_source": prompt_template,
                "template_variables": template_variables
            },
            "grammar_generator": {
                "model": model
            }
        })["output_parser"]["model"]
    

grammar_pipeline = _GrammarPipeline()