import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import gradio as gr

# Load .env file if present (important for local dev, optional on Render)
load_dotenv()

# Ensure the environment variable is set
if not os.getenv("HF_API_KEY"):
    raise ValueError("HF_API_KEY environment variable is not set.")

# Setup HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_API_KEY")
)

model = ChatHuggingFace(llm=llm)

# Chat function
def chat(message, history):
    response = model.invoke(message)
    return response.content

# Get PORT from Render environment or use 7860 as fallback
port = int(os.environ.get("PORT", 7860))

# Launch Gradio app with correct host and port
gr.ChatInterface(fn=chat, type="messages").launch(server_name="0.0.0.0", server_port=port)
