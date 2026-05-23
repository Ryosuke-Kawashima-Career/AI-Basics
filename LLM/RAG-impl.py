import sys
import re
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Fetch course pages from the Matsuo Lab lecture list website
def fetch_webpage(url: str = "https://weblab.t.u-tokyo.ac.jp/lecture/course-list/") -> List[Dict[str, str]]:
    res = requests.get(url)
    res.raise_for_status()
    # Correct encoding for Japanese characters
    res.encoding = res.apparent_encoding
    
    soup = BeautifulSoup(res.text, "html.parser")
    results = []
    
    # Look for anchor tags targeting external sites/new tabs
    for element in soup.find_all("a"):
        if element.get("target") == "_blank" and element.get("href"):
            # Fall back to text content if 'title' attribute is empty
            title = element.get("title") or element.get_text(strip=True) or "No Title"
            results.append({
                "title": title,
                "url": element.get("href"),
                "content": element.get_text(strip=True)
            })
    
    print(f"Fetched {len(results)} items from webpage.")
    if results:
        print("Sample:", results[:2])
    return results

# Chunking the scraped data into manageable text snippets
def chunking_fixed_length(data: List[Dict[str, str]], fix_len: int = 300) -> List[str]:
    chunks = []
    for webpage in data:
        title = webpage.get("title") or "No Title"
        url = webpage.get("url") or "No URL"
        content = webpage.get("content") or ""
        
        chunk_text = f"title: {title}\nurl: {url}\ncontent: {content}"
        
        # Split text on Japanese punctuation or whitespace to avoid cutting sentences mid-way
        sentences = re.split(r'(?<=[。\n])| +', chunk_text)
        
        fixed_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # If adding the sentence exceeds target length, save current chunk and start new one
            if len(fixed_chunk) + len(sentence) > fix_len and fixed_chunk:
                chunks.append(fixed_chunk.strip())
                fixed_chunk = sentence
            else:
                if fixed_chunk:
                    fixed_chunk += " " + sentence
                else:
                    fixed_chunk = sentence
        
        if fixed_chunk:
            chunks.append(fixed_chunk.strip())
            
    return chunks

# Normalize embedding vector
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    norm = np.clip(norm, a_min=1e-10, a_max=None)  # Avoid division by zero
    return vec / norm

# Retrieve top K similar chunks from the vector space
def top_k_retrieval(embed_model, query: str, chunked_data: List[str], k: int = 5) -> List[str]:
    query_embedding = embed_model.encode(query)
    chunk_embeddings = embed_model.encode(chunked_data)
    
    # Normalize the embeddings to unit vectors for cosine similarity
    query_embedding = normalize_vector(query_embedding)
    chunk_embeddings = np.array([normalize_vector(embedding) for embedding in chunk_embeddings])
    
    # Cosine similarity is the dot product of normalized vectors
    cosine_similarities = query_embedding @ chunk_embeddings.T
    
    # Get indices of top k elements in descending order
    top_k_indices = cosine_similarities.argsort()[::-1][:k]
    top_k_chunks = [chunked_data[index] for index in top_k_indices]
    return top_k_chunks

# Combine user query and retrieved context
def integrate_query_and_retrieved(query: str, retrieved_chunks: List[str]) -> str:
    integrated_prompt = f"Query: {query}\n\nRetrieved Information:\n" + "\n---\n".join(retrieved_chunks)
    return integrated_prompt

# Feed final prompt to LLM and generate answer
def generate_response(llm, tokenizer, integrated_prompt):
    messages = [
        {"role": "user", "content": integrated_prompt},
    ]
    
    # Return prompt as string to be processed by tokenizer
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    input_tokens = tokenizer(prompt_str, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        generated_tokens = llm.generate(
            **input_tokens,
            max_new_tokens=256,
            do_sample=False  # Deterministic output for testing
        )
    
    # Extract only the newly generated tokens
    prompt_len = input_tokens["input_ids"].shape[1]
    output_ids = generated_tokens[0][prompt_len:].tolist()
    
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content

# Quick query test
def simple_query(llm, tokenizer):
    prompt = "Who is the prime minister of India?"
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    input_tokens = tokenizer(prompt_str, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        generated_tokens = llm.generate(
            **input_tokens,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )
        
    prompt_len = input_tokens["input_ids"].shape[1]
    output_ids = generated_tokens[0][prompt_len:].tolist()
    decoded_response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("Simple Query Response:", decoded_response)

def main():
    # CPU-friendly and highly efficient model setup
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading LLM model: {model_name}...")
    
    # Standard config, auto device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Fetching webpage data...")
    data = fetch_webpage()
    if not data:
        print("Error: No data fetched from webpage. Vector store cannot be constructed.")
        return
        
    print("Chunking data...")
    chunks = chunking_fixed_length(data, fix_len=300)
    print(f"Created {len(chunks)} text chunks.")
    
    # Using a fast, lightweight multilingual model for embeddings
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading embedding model: {embed_model_name}...")
    embed_model = SentenceTransformer(embed_model_name)
    
    query = "What courses are available in Tokyo University Matsuo Lab?"
    print(f"\nUser Query: '{query}'")
    
    print("Retrieving matched contexts...")
    retrieved_chunks = top_k_retrieval(embed_model, query, chunks, k=3)
    
    print("\n--- Retrieved Top Chunks ---")
    for idx, c in enumerate(retrieved_chunks):
        print(f"Chunk {idx+1}:\n{c}\n")
    
    print("Integrating context and query...")
    integrated_prompt = integrate_query_and_retrieved(query, retrieved_chunks)
    
    print("Generating response from LLM...")
    response = generate_response(model, tokenizer, integrated_prompt)
    print("\n=== Final LLM Response ===")
    print(response)
    print("==========================")
    
    print("\nRunning simple query test...")
    simple_query(model, tokenizer)

if __name__ == "__main__":
    main()
