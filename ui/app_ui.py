import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"

def upload_file(file):
    if file is None:
        return "Please upload a file."

    files = {"file": open(file.name, "rb")}
    response = requests.post(f"{API_URL}/upload", files=files)

    return response.json()["message"]

def chat_fn(message, history):
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": message,
            "history": []
        }
    )

    return response.json()["answer"]


with gr.Blocks() as demo:

    gr.Markdown("# Smart Contract Assistant")

    with gr.Tab("Upload Document"):
        file_input = gr.File()
        upload_button = gr.Button("Process Document")
        upload_output = gr.Textbox()

        upload_button.click(
            upload_file,
            inputs=file_input,
            outputs=upload_output
        )

    with gr.Tab("Ask Questions"):
        gr.ChatInterface(fn=chat_fn)

demo.launch()