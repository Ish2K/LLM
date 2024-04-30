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

dataset = load_dataset("./data", split="train", trust_remote_code=True)

print(dataset)
# create a trainer specific for language modeling
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenizer(dataset["prompt"], return_tensors="pt"),
    eval_dataset=tokenizer(dataset["completion"], return_tensors="pt")
)

# start training
trainer.train()