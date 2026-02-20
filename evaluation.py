from app.rag_pipeline import SmartContractAssistant
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

assistant = SmartContractAssistant()

assistant.ingest_document("data/My_Learning_NVIDIA.pdf")

test_cases = [
    {
        "question": "Who is the certificate awarded to?",
        "expected_answer": "Kareem"
    },
    {
        "question": "What is the date of the certificate?",
        "expected_answer": "February 15"
    }
]

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

correct_answers = 0
retrieval_scores = []
grounding_scores = []

for case in test_cases:

    print("\n==============================")
    print("Question:", case["question"])

    answer = assistant.ask(case["question"])
    print("Model Answer:", answer)

    # Accuracy
    if case["expected_answer"].lower() in answer.lower():
        correct_answers += 1
        print("Correct")
    else:
        print("Incorrect")

    # Similarity
    question_emb = embedding_model.encode(case["question"])
    answer_emb = embedding_model.encode(answer)

    similarity = cosine_similarity(
        [question_emb],
        [answer_emb]
    )[0][0]

    retrieval_scores.append(similarity)
    print("Similarity:", round(similarity, 3))

    grounded = similarity > 0.5
    grounding_scores.append(int(grounded))
    print("Grounded:", grounded)

accuracy = correct_answers / len(test_cases)
avg_similarity = np.mean(retrieval_scores)
grounding_rate = np.mean(grounding_scores)

print("\n==============================")
print("FINAL RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Average Similarity:", round(avg_similarity, 3))
print("Grounding Rate:", round(grounding_rate * 100, 2), "%")
