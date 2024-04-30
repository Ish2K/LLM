import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# build chatbot to summarize text

def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

iface = gr.Interface(fn=summarize_text, inputs="text", outputs="text")

iface.launch()