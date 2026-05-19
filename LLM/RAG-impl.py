# Import necessary libraries
import sys
import transformers

# Monkey-patch DisjunctiveConstraint for transformers_stream_generator compatibility with transformers v5+
if not hasattr(transformers, "DisjunctiveConstraint"):
    try:
        from transformers.generation import DisjunctiveConstraint
        transformers.DisjunctiveConstraint = DisjunctiveConstraint
    except ImportError:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

# Model loading
model_name = "Qwen/Qwen-7B-Chat"
model_config = {
    "trust_remote_code": True,
    "low_cpu_mem_usage": True
}
### Expand the model config in **model_config** to load the model
model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
## Embedding model loading
tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)

# Data loader (by request and Beautiful soup)
import requests
from bs4 import BeautifulSoup
import re
def fetch_webpage(url: str = "https://weblab.t.u-tokyo.ac.jp/lecture/course-list/") -> list[dict[str, str]]:
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    results = []
    for element in soup.find_all():
        if element.get("style") is not None and element.get("target") == "_blank":
            results.append({
                "title": element.get("title"),
                "url": element.get("href"),
                "content": element.get_text(strip=True)
            })
    print(results[:3])
    return results

## Chunking of the data
from typing import List, Dict
import tiktoken
def chunking_fixed_length(data: List[Dict[str, str]], fix_len: int=300) -> List[str]:
    chunks = []
    for webpage in data:
        chunk = "title: " + webpage["title"] + "\n" + "url: " + webpage["url"] + "\n" + "content: " + webpage["content"]
        sentences = chunk.split("。").split("\n").split(" ")
        fixed_chunk = ""
        for i, sentence in enumerate(sentences):
            fixed_chunk += sentence + ""
            if len(fixed_chunk) > fix_len:
                chunks.append(fixed_chunk)
                fixed_chunk = ""
        if fixed_chunk: # Add the last chunk if it's not empty
            chunks.append(fixed_chunk)

    return chunks

chuncked_data = chunking_fixed_length(data, fix_len=300)
print(chuncked_data)

## Create a Vector Store

# Build a pipleline for the RAG system
def simple_query(llm, tokenizer):
    prompt = "Who is the prime minister of India?"
    messages = [
        {"role": "user", "content": prompt},
    ]
    ## Get the user query and then, **Embed** it
    tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True
    )
    input_tokens = tokenizer(tokens, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(**input_tokens, do_sample=True)
    decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print("Decoded Tokens:", decoded_tokens[0])

## Retrieve the Top K contents related to the query from the Vector Store

## Integrate the query and the retrieved contents to form a prompt

# Feed it to the **LLM**

# Return the response

# Evaluate and Test the RAG system
def main():
    simple_query(model, tokenizer)

if __name__ == "__main__":
    main()
