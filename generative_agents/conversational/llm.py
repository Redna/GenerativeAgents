import os

from transformers import AutoTokenizer, pipeline

from langchain.llms import HuggingFacePipeline
import torch

model_checkpoint = "HuggingFaceH4/zephyr-7b-alpha"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pipe = pipeline(
    "text-generation",
    model=model_checkpoint,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_new_tokens=250,
    temperature=0.0,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)


if __name__ == "__main__":
    print(llm.generate("Hello world!"))