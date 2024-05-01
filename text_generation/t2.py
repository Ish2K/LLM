from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

print("1")
checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("2")
# You may want to use bfloat16 and/or move to GPU here
model = AutoModelForCausalLM.from_pretrained(checkpoint, use_bfloat16=True)


dataset = load_dataset("json", data_files = "./data/data1.jsonl", split="train", trust_remote_code=True)

print("2")
print(dataset)
tokenized_chat = tokenizer.apply_chat_template(dataset, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))