from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

## Sentiment Analysis
classifier = pipeline('sentiment-analysis')
result = classifier('We are very happy to show you the 🤗 Transformers library.'
                    )

print(result)

### Sentiment Analysis with a model from the model hub

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

result = classifier('We are very happy to show you the 🤗 Transformers library.'
)

print(result)

## Dry run of tokenization

sequence = "We are very happy to show you the 🤗 Transformers library."
res = tokenizer(sequence)
print(res)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded = tokenizer.decode(ids)
print(decoded)

# Fine-tuning a model on a text classification task

X_train = ["I love you", "I hate you", "You are awesome", "You are terrible"]
y_train = [1, 0, 1, 0]

res = classifier(X_train)
print(res)

batch = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    logits = outputs.logits
    predictions = F.softmax(logits, dim=-1)
    print(predictions)
    labels = torch.tensor(y_train).unsqueeze(0)
    loss = F.cross_entropy(logits, labels)
    print(loss)

