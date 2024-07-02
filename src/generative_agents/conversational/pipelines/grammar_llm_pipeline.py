import json
import os
import hashlib
import pickle
from typing import Any, Dict, List, Optional

from colorama import Back, Fore, Style
from haystack import Pipeline, component
from haystack.core.component import Component
from haystack.components.builders import DynamicPromptBuilder, DynamicChatPromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.generators.openai import OpenAIGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

import instructor
from llama_cpp import LlamaGrammar
from openai import BadRequestError
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json


from groq import Groq

from generative_agents import global_state
from generative_agents.utils import colored, generate_tick_hash_from_signature

import contextvars

last_cache_file = contextvars.ContextVar("last_cache_file")

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


@component
class PydanticToJSONSchema:
    @component.output_types(schema=str)
    def run(self, model: BaseModel):
        return {"schema": json.dumps(model.model_json_schema(), indent=4)}



@component
class GroqInstrcutorGenerator:

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("GROQ_API_KEY"),
        model: str = "gpt-3.5-turbo",
    ):
        client = Groq(
            api_key=api_key._token,
        )

        self.client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

        self.model = model

    @component.output_types(model=BaseModel)
    def run(self, messages: List[ChatMessage], pydantic_model: BaseModel, generation_kwargs: Optional[Dict[str, Any]] = None):
        try:
            response_model = self.client.chat.completions.create(
                model=self.model,
                response_model=pydantic_model,
                messages=[message.to_openai_format() for message in messages]
            )
        except Exception as e:
            raise BadRequestError(f"Error generating response: {e}")

        return {"model": response_model}


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

        json_result = from_json(replies[0])
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
        hashable_kwargs = {k: str(v) for k, v in kwargs.items()}

        hash_key = generate_tick_hash_from_signature(**hashable_kwargs)
        cache_dir = f".generation_cache/llm/tick_{global_state.tick}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = f"{cache_dir}/{hash_key}.json"

        last_cache_file.set(cache_file_path)
 
        with colored(Style.BRIGHT, Fore.CYAN, Back.BLACK):
            print(kwargs[self.input_name])

        if os.path.exists(cache_file_path):
            output = dict()
            cached_output = json.load(open(cache_file_path, "r"))

            if "pydantic_model" in kwargs:
                pydantic_model = kwargs["pydantic_model"]
                output[self.output_name] = pydantic_model(**json.loads(cached_output))
        else:
            output = self.component.run(**kwargs)

            if isinstance(output[self.output_name], BaseModel):
                json.dump(output[self.output_name].model_dump_json(), open(cache_file_path, "w"), indent=4)
            else:
                out = output[self.output_name][-1] if isinstance(output[self.output_name], list) else output[self.output_name]
                json.dump(out, open(cache_file_path, "w"), indent=4)

        with colored(Style.BRIGHT, Fore.GREEN, Back.BLACK):
            if isinstance(output[self.output_name], BaseModel):
                print(json.dumps(output[self.output_name].model_dump_json(), indent=4))
            else:
                out = output[self.output_name][-1] if isinstance(output[self.output_name], list) else output[self.output_name]
                print(json.dumps(json.loads(out), indent=4))

        return output

class _GrammarPipeline:
    def __init__(self):
        self.pipe = Pipeline()

        # print current working directory
        print(os.getcwd())

        generator = OpenAIGenerator(
            api_key=Secret.from_token("<API_KEY>"),
            model="llama3-8b-8192", #"models/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
            api_base_url="https://api.groq.com/openai/v1", #"http://localhost:30091/v1/",
            generation_kwargs={
                "max_tokens": 4096,
                "temperature": 0.6
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
        #prompt_template += "\n\n### Answer in valid JSON. Output hint:\n" + get_output_hint(model) + "\n###"
        
        prompt_template += "\n\n### Answer in valid JSON. Output hint:\n" + json.dumps(model.model_json_schema(), indent=2) + "\n###"

        generation_kwargs = {
            "response_format": {
                "type": "json_object",
                "schema": model.model_json_schema()
            }
        }

        try:
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
        except ValidationError as e:
            print(f"Error: {e}")
            #delete the cache file
            os.remove(last_cache_file.get())
            raise e
        except BadRequestError as e:
            print(f"Error: {e}")
            raise e

        return output


class _GroqGrammarPipeline:
    def __init__(self):
        self.pipe = Pipeline()

        # print current working directory
        print(os.getcwd())

        generator = GroqInstrcutorGenerator(
            api_key=Secret.from_token("<api_key>"),
            model="llama3-8b-8192"
        )

        printable = PrintableGenerator(generator, "messages", "model")

        self.pipe.add_component("prompt", instance=DynamicChatPromptBuilder())
        self.pipe.add_component("llm", printable)

        self.pipe.connect("prompt.prompt", "llm.messages")

    def run(
        self, model: BaseModel, prompt_template: str, template_variables: dict[str, any]
    ): 
        #prompt_template += "\n\n### Answer in valid JSON. Output hint:\n" + get_output_hint(model) + "\n###"
        
        messages = [ChatMessage.from_system("Follow the task as closely as possible. Answer in valid JSON. Output hint:\n" + get_output_hint(model)),
                    ChatMessage.from_user(prompt_template)]

        try:
            output = self.pipe.run(data={
                    "prompt": {
                        "prompt_source": messages,
                        "template_variables": template_variables
                    },
                    "llm": {
                        "pydantic_model": model
                    }
                }
            )["llm"]["model"]
        except ValidationError as e:
            print(f"Error: {e}")
            #delete the cache file
            os.remove(last_cache_file.get())
            raise e
        except BadRequestError as e:
            print(f"Error: {e}")
            raise e

        return output

grammar_pipeline = _GroqGrammarPipeline()
