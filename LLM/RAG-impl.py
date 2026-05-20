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
## Embedding is need for verctor search
from sentence_transformers import SentenceTransformer
import numpy as np
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    norm = np.clip(norm, a_min=1e-10, a_max=None)  # Avoid division by zero
    return vec / norm
embed_model_name = "Qwen/Qwen-7B-Chat"
embed_model = SentenceTransformer(embed_model_name)
def top_k_retrieval(query: str, chuncked_data: List[str], k: int = 5) -> List[str]:
    query_embedding = embed_model.encode(query)
    chunk_embeddings = embed_model.encode(chuncked_data)
    # Normalize the embeddings to unit vectors for cosine similarity
    query_embedding = normalize_vector(query_embedding)
    chunk_embeddings = np.array([normalize_vector(embedding) for embedding in chunk_embeddings])
    cosine_similarities = query_embedding @ chunk_embeddings.T
    top_k_indices = cosine_similarities.argsort()[::-1][:k]
    ### argsort returns the indices of the original array that would sort the array. [::-1] reverses the order to get the top k highest similarities, and [:k] selects the top k indices.
    top_k_chunks = [chuncked_data[index] for index in top_k_indices]
    return top_k_chunks

## Integrate the query and the retrieved contents to form a prompt
def integrate_query_and_retrieved(query: str, retrieved_chunks: List[str]) -> str:
    integrated_prompt = query + "\n\n" + "Retrieved Information:\n" + "\n".join(retrieved_chunks)
    return integrated_prompt

# Feed it to the **LLM**
def generate_response(llm, tokenizer, integrated_prompt):
    message = [
        {"role": "user", "content": integrated_prompt},
    ]
    tokens = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True
    )
    # Embedding the integrated prompt
    input_tokens = tokenizer(tokens, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(**input_tokens)
    ## Q. What is the data structure of generated_tokens? How to decode it to text?
    output_ids = generated_tokens[0][input_tokens["input_ids"].shape[1]:].tolist()
    # Vector to text
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content

# Evaluate and Test the RAG system
def main():
    simple_query(model, tokenizer)

if __name__ == "__main__":
    main()
