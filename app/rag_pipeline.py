import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartContractAssistant:

    def __init__(self):

        print(" Loading embedding model")
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print(" Loading FLAN-T5 QA model")
        self.generator = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base"
        )

        self.client = chromadb.Client(
            Settings(
                persist_directory="./chroma_db",
                is_persistent=True
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="smart_contract_docs"
        )

        self.chat_history = []

    def ingest_document(self, pdf_path):

        try:
            self.collection.delete(where={"ids": {"$ne": "dummy"}})
        except:
            pass

        doc = fitz.open(pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk).tolist()
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"chunk_{i}"]
            )

        self.chat_history = []

        return len(chunks)

    def ask(self, question):

        query_embedding = self.embedding_model.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        if not results["documents"] or not results["documents"][0]:
            return "Not found in document."

        context_chunks = results["documents"][0]
        context = "\n\n".join(context_chunks)

        history_text = ""
        for q, a in self.chat_history:
            history_text += f"User: {q}\nAssistant: {a}\n"

        prompt = f"""
You are a strict contract assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not clearly in the context, say: Not found in document.
- Answer in 1–2 sentences only.
- Do NOT repeat the question.

Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.generator(
            prompt,
            max_new_tokens=120
        )

        answer = response[0]["generated_text"].strip()

        grounded = any(answer.lower() in chunk.lower() for chunk in context_chunks)

        if not grounded:
            answer = "Not found in document."

        self.chat_history.append((question, answer))

        return answer
