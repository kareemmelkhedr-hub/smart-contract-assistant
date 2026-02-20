# Smart Contract Assistant

RAG-based Question Answering system for Smart Contracts.

## Architecture
- FastAPI Backend
- Gradio Frontend
- LangChain
- ChromaDB
- Sentence Transformers
- FLAN-T5

## Features
- Upload PDF
- Ask Questions
- Context-aware answers
- Evaluation metrics

## How to Run

1. Create virtual environment:
   python -m venv venv

2. Activate:
   venv\Scripts\activate

3. Install requirements:
   pip install -r requirements.txt

4. Run Backend:
   uvicorn backend.api:app --reload

5. Run Frontend:
   python -m ui.app_ui