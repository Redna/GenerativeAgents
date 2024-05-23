import json
import os
import hashlib
import pickle

from haystack import Pipeline, component
from haystack.components.builders import DynamicPromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator


from llama_cpp import LlamaGrammar
from pydantic import BaseModel
from pydantic_core import from_json

from generative_agents import global_state


def pydantic_to_grammar(model: BaseModel) -> LlamaGrammar:
    schema = model.model_json_schema()

    return LlamaGrammar.from_json_schema(schema)


def escape_json_string(input_str):
    escaped_str = (
        input_str.replace("\n", "\\n").replace("\r", "\\r")
    )
    return escaped_str

@component
class PydanticToJSONSchema:
    @component.output_types(schema=str)
    def run(self, model: BaseModel):
        return {"schema": json.dumps(model.model_json_schema(), indent=4)}

@component
class GrammarGenerator:
    @component.output_types(generation_kwargs=dict[str, any])
    def run(self, schema: str):
        return {"generation_kwargs": {"grammar": LlamaGrammar.from_json_schema(schema)}}


@component
class LLMOutputParser:
    @component.output_types(model=BaseModel)
    def run(self, model: BaseModel, replies: list[str]):

        json_result = from_json(escape_json_string(replies[0]))
        for key, value in json_result.items():
            json_result[key] = value.strip() if isinstance(value, str) else value

        return {"model": model(**json_result)}


class _GrammarPipeline:
    def __init__(self):
        self.pipe = Pipeline()

        # print current working directory
        print(os.getcwd())

        generator = LlamaCppGenerator(
            model="models/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
            n_ctx=8096,
            n_batch=512,
            model_kwargs={
                "n_gpu_layers": -1,
                "verbose": True
            },
            generation_kwargs={
                "max_tokens": 4096,
                "temperature": 1,
            }
        )
        self.pipe.add_component("prompt", instance=DynamicPromptBuilder())

        self.pipe.add_component("grammar_generator", GrammarGenerator())
        self.pipe.add_component("llm", generator)
        self.pipe.add_component("output_parser", LLMOutputParser())

        self.pipe.connect("prompt.prompt", "llm.prompt")
        self.pipe.connect("grammar_generator", "llm.generation_kwargs")
        self.pipe.connect("llm.replies", "output_parser.replies")

    def run(
        self, model: BaseModel, prompt_template: str, template_variables: dict[str, any]
    ): 

        hash_key = hashlib.sha256((model.__name__ + prompt_template + json.dumps(template_variables) + str(global_state.tick)).encode()).hexdigest()

        cache_dir = f".generation_cache/tick_{global_state.tick}"
        os.makedirs(cache_dir, exist_ok=True)

        cache_file_path = f"{cache_dir}/{hash_key}.json"

        if os.path.exists(cache_file_path):
            with open(cache_file_path, "rb") as cache_file:
                return model.model_validate_json(cache_file.read())

        schema = json.dumps(model.model_json_schema(), indent=4)

        prompt_template += "\n\n### JSON Schema to use for the output:\n" + schema + "\n###"
        output = self.pipe.run(data={
                "prompt": {
                    "prompt_source": prompt_template,
                    "template_variables": template_variables,
                },
                "grammar_generator": {"schema": schema},
                "output_parser": {"model": model}
            }
        )["output_parser"]["model"]

        json.dump(output.model_dump(mode='json'), open(cache_file_path, "w"), indent=4)

        return output


grammar_pipeline = _GrammarPipeline()
