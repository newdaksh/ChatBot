import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import gradio as gr
# Ensure the environment variable is set
if not os.getenv("HF_API_KEY"):
    raise ValueError("HF_API_KEY environment variable is not set.")

# Setup HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HF_API_KEY")
)

model = ChatHuggingFace(llm=llm)

# Chat function
def chat(message, history):
    response = model.invoke(message)
    return response.content

# Launch Gradio Chat UI
gr.ChatInterface(fn=chat, type="messages").launch()