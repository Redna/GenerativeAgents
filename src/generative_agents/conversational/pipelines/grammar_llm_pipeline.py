import json
import os
import hashlib
import pickle

from colorama import Back, Fore, Style
from haystack import Pipeline, component
from haystack.core.component import Component
from haystack.components.builders import DynamicPromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.generators.openai import OpenAIGenerator
from haystack.utils import Secret

from llama_cpp import LlamaGrammar
from pydantic import BaseModel
from pydantic_core import from_json

from generative_agents import global_state
from generative_agents.utils import colored, generate_tick_hash_from_signature


def get_output_hint(model: BaseModel, indent: int=2) -> dict[str, dict[str, any]]:
    schema = model.model_json_schema()

    def _get_field_definitions(sub_schema: dict[str, any], new_indent: int):
        fields = sub_schema.get('properties', {})
        indent_str = " " * new_indent
        easy_string = f"{indent_str}{{\n"
        
        for field_name, details in fields.items():
            field_name_indent = " " * (new_indent + indent)
            field_name_str = f'{field_name_indent}"{field_name}"'
            description_str = details.get('description') if details.get('description') else field_name
            _type = details.get('type')

            def _get_ref_schema(ref):
                ref_name = ref.split('/')[-1]
                ref_schema = schema['$defs'][ref_name]
                return ref_schema
            
            if not _type:
                ref = details.get('$ref')
                if not ref:
                    ref = details.get('allOf', [{}])[0].get('$ref')

                reference_schema = _get_ref_schema(ref)
                if reference_schema["type"] not in ['object', 'array']:
                    if "const" in reference_schema:
                        description_str = f"Value: {reference_schema['const']}"
                    elif "enum" in reference_schema:
                        description_str += f" | Possible Values: {reference_schema['enum']}"
                    easy_string += f"{field_name_str}: # {description_str} | Datatype: [{reference_schema['type']}]\n"
                else:
                    easy_string += f"{field_name_str}: # {description_str} | Datatype: [object]\n"
                    easy_string += _get_field_definitions(_get_ref_schema(ref), new_indent=new_indent + indent)
            elif _type == 'array' and details.get('items', {}).get('$ref'):
                easy_string += f"{field_name_str}: # {description_str} | Datatype: [{_type}]\n"
                ref = details['items']['$ref']
                easy_string += _get_field_definitions(_get_ref_schema(ref), new_indent=new_indent + indent)

            else:
                if "const" in details:
                    description_str = f"Value: {details['const']}"
                easy_string += f"{field_name_str}: # {description_str} | Datatype: [{_type}]\n"

        easy_string += f"{indent_str}}}\n"
        return easy_string
    
    return _get_field_definitions(schema, new_indent=indent)


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
        return {"generation_kwargs": {
                    "extra_body": {
                        "grammar": LlamaGrammar.from_json_schema(schema)
                    }    
                }}



@component
class LLMOutputParser:
    @component.output_types(model=BaseModel)
    def run(self, model: BaseModel, replies: list[str]):

        json_result = from_json(escape_json_string(replies[0]))
        for key, value in json_result.items():
            json_result[key] = value.strip() if isinstance(value, str) else value

        return {"model": model(**json_result)}

@component
class PrintableGenerator:
    def __init__(self, c: Component, input_name: str, output_name: str):
        self.component = c
        self.__haystack_input__ = c.__haystack_input__
        self.__haystack_output__ = c.__haystack_output__
        self.input_name = input_name
        self.output_name = output_name
        if hasattr(c, "warm_up"):
            c.warm_up()

    def run(self, **kwargs):
        hash_key = generate_tick_hash_from_signature(**kwargs)
        cache_dir = f".generation_cache/llm/tick_{global_state.tick}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = f"{cache_dir}/{hash_key}.json"
 
        with colored(Style.BRIGHT, Fore.CYAN, Back.BLACK):
            print(kwargs[self.input_name])

        if os.path.exists(cache_file_path):
            output = json.load(open(cache_file_path, "r"))
        else:
            output = self.component.run(**kwargs)
            json.dump(output, open(cache_file_path, "w"), indent=4)

        with colored(Style.BRIGHT, Fore.GREEN, Back.BLACK):
            out = output[self.output_name][-1] if isinstance(output[self.output_name], list) else output[self.output_name]
            print(json.dumps(json.loads(out), indent=4))

        return output

class _GrammarPipeline:
    def __init__(self):
        self.pipe = Pipeline()

        # print current working directory
        print(os.getcwd())

        generator = OpenAIGenerator(
            api_key=Secret.from_token("secret"),
            model="models/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
            api_base_url="http://localhost:30091/v1/",
            generation_kwargs={
                "max_tokens": 4096,
                "temperature": 0.8,
                "top_p": 0.8
            }
        )

        printable = PrintableGenerator(generator, "prompt", "replies")

        self.pipe.add_component("prompt", instance=DynamicPromptBuilder())
        self.pipe.add_component("llm", printable)
        self.pipe.add_component("output_parser", LLMOutputParser())

        self.pipe.connect("prompt.prompt", "llm.prompt")
        self.pipe.connect("llm.replies", "output_parser.replies")

    def run(
        self, model: BaseModel, prompt_template: str, template_variables: dict[str, any]
    ): 
        prompt_template += "\n\n### Answer in valid JSON. Output hint:\n" + get_output_hint(model) + "\n###"

        generation_kwargs = {
            "response_format": {
                "type": "json_object",
                "schema": model.model_json_schema()
            }
        }

        output = self.pipe.run(data={
                "prompt": {
                    "prompt_source": prompt_template,
                    "template_variables": template_variables,
                },
                "llm": {
                    "generation_kwargs": generation_kwargs
                },
                "output_parser": {"model": model}
            }
        )["output_parser"]["model"]

        return output


grammar_pipeline = _GrammarPipeline()
