from transformers import pipeline

classifier = pipeline('zero-shot-classification')

result = classifier('This is a course about the Transformers library',
                    candidate_labels=['education', 'politics', 'business'],
                    multi_class=True)

print(result)