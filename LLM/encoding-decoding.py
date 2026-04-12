from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt-5-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prompt adopts XML style
prompt = """
You are a professional cricket consultant in New Delhi, India.
Your task is to answer the user's query about India. The objective is to help the user make Indian friends through conversations about cricket.
The query is as follows:
<query>
{content}
</query>
The format uses a bilingual notation with English primary and Japanese secondary as notations for keywords.
"""

# JSON is the defacto format of data exchange between programs
messages = [
    {"role": "system", "content": prompt.format(content="What is the capital of India?")},
]

tokens = tokenizers.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# Training mode
with torch.no_grad():
    output = model.generate(**model_inputs, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
