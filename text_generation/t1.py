from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Let's chat for 5 lines
# for step in range(5):
#     # encode the new user input, add the eos_token and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#     # generated a response while limiting the total chat history to 1000 tokens, 
#     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#     # pretty print last ouput tokens from bot
#     print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

# fine tune the model

from transformers import Trainer, TrainingArguments
from trl import SFTTrainer

# define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=100,
    save_total_limit=5,
    overwrite_output_dir=True,
    do_train=True,
    output_dir='./results'
)

dataset = load_dataset("json", data_files = "./data/data.jsonl", split="train", trust_remote_code=True)

print(dataset)
# create a trainer specific for language modeling
trainer = SFTTrainer(
    "facebook/opt-350m",
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="completion",
    packing=True,
)

# start training
trainer.train()

# generate inference

from transformers import pipeline

# load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./results")

# load the tokenizer
