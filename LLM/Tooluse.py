import sys
import re
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

class TooluseLLM:
    def __init__(self, model_name: str):
        """Initiates the Tool-use LLM system"""
        self.model_name = model_name
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = "auto",
            torch_dtype = "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def fetch_webpage(self, url: str) -> List[str]:
        """Fetch data from a given URL
        Args:
            url (str): URL to fetch data from
        Returns:
            List[str]: List of data fetched from the URL
        """
        res = requests.get(url)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "html.parser")
        results = []

        for element in soup.find_all(["p", "div", "span"]):
            text = element.get_text(strip=True)
            if text and text not in results:
                results.append(text)
        return results[:10]  # Limit to top 10 elements to avoid context overflow
    
    def calculate_sum(self, temperatures: List[float]) -> float:
        """Calculate sum of temperatures
        Args:
            temperatures (List[float]): List of temperatures
        Returns:
            float: Sum of temperatures
        """
        # Safely cast components to float in case the LLM passes strings
        return sum(float(t) for t in temperatures)
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response to the conversation history
        Args:
            messages (List[Dict[str, str]]): Complete conversation history
        Returns:
            str: Next generated response
        """
        prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_tokens = self.tokenizer(prompt_str, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            generated_tokens = self.llm.generate(
                **input_tokens,
                max_new_tokens=256,
                do_sample=False,
            )
        sequence_length = input_tokens["input_ids"].shape[1]
        output_ids = generated_tokens[0][sequence_length:].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content
    
    def tool_call(self, output: str) -> Dict[str, str]:
        """Based on the output of LLM, it get right tools"""
        try:
            tool_call_str = re.search(f"<tool_call>(.+?)</tool_call>", output).group(1)
            tool_call = json.loads(tool_call_str)
            function_name = tool_call.get('name', '')
            arguments = tool_call.get('args', '')
            response = ""
            if function_name == "fetch_webpage":
                response = self.fetch_webpage(arguments)
            elif function_name == "calculate_sum":
                response = self.calculate_sum(arguments)
            else:
                response = "Unknown tool"
            return {"tool_response": f"{function_name} with args: {arguments} result: {response}"}
        except Exception as e:
            return {"tool_response": f"Error parsing tool call: {str(e)}"}


    
    def iterative_tool_use(self, query: str, max_loop: int = 3) -> str:
        """It use tools iteratively to complete the query"""
        system_prompt = (
            "You are a helpful assistant that can use tools to fetch data and perform calculations. "
            "When you need to use a tool, respond with a <tool_call> tag containing a JSON object with 'name' and 'args'. "
            "For example: <tool_call>{\"name\": \"fetch_webpage\", \"args\": {\"url\": \"http://example.com\"}}</tool_call>. "
            "After receiving the tool response, you can continue to generate more tool calls or provide the final answer. "
            "Available tools: fetch_webpage(url: str), calculate_sum(temperatures: list of floats)"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        outputs = []
        tool_responses = []
        
        for _trial in range(max_loop):
            print(f"Iteration {_trial + 1}:")
            output = self.generate_response(messages)
            outputs.append(output)
            messages.append({"role": "assistant", "content": output})
            print(f"LLM Output: {output}")
            
            tool_response = self.tool_call(output)
            if not tool_response:
                break # Exit the loop if no tool calls are made (assumes final answer reached)
                
            print(f"Tool Response: {tool_response}")
            tool_responses.append(tool_response)
            messages.append({"role": "user", "content": f"Tool response: {tool_response['tool_response']}"})
            
        result = f"Final Answer: {outputs[-1]}\nTool Responses: {tool_responses}"
        return result

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Switched to a non-gated instruction model
    tooluse_llm = TooluseLLM(model_name)
    print("Model loaded successfully.")
    query = "What is the sum of the maximum temperatures in New York and California in April, 2026?"
    print(f"Query: {query}")
    response = tooluse_llm.iterative_tool_use(query)
    print(response)
    
if __name__ == "__main__":
    main()
