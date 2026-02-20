from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import shutil
import os

from app.rag_pipeline import SmartContractAssistant

app = FastAPI()

assistant = SmartContractAssistant()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)



class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []



@app.get("/")
def root():
    return {"message": "Smart Contract Assistant API is running "}



@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = assistant.ingest_document(file_path)

    return {
        "message": "Document processed successfully",
        "chunks": chunks
    }



@app.post("/ask")
async def ask_question(request: ChatRequest):

    # restore history
    assistant.chat_history = [
        (msg.content, "")
        for msg in request.history
        if msg.role == "user"
    ]

    answer = assistant.ask(request.question)

    return {
        "answer": answer
    }