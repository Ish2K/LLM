from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

result = generator('In this tutorial, we will learn how to',
                    max_length=50,
                    num_return_sequences=3)

print(result)