import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

print("Loading embedding model")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading LLM")
generator = pipeline("text-generation", model="google/flan-t5-base")
print("Models loaded!")

pdf_path = "data/My Learning _ NVIDIA.pdf"
doc = fitz.open(pdf_path)

full_text = ""
for page in doc:
    full_text += page.get_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(full_text)

client = chromadb.Client()
collection = client.create_collection(name="contract_chunks")

for i, chunk in enumerate(chunks):
    embedding = embedding_model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )


query = input("\n Ask a question about the document: ")

query_embedding = embedding_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

context = results["documents"][0][0]

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

response = generator(prompt, max_length=200, do_sample=False)

print("\n Answer:\n")
print(response[0]["generated_text"])
