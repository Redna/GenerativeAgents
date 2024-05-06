

from llama_cpp import LlamaGrammar
from pydantic import BaseModel


def pydantic_to_grammar(model: BaseModel) -> LlamaGrammar:
    schema = model.model_json_schema()

    return LlamaGrammar.from_json_schema(schema)