
import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer


def extract_text(pdf_path):
    document = pymupdf.open(pdf_path)
    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def search_chunks(question, chunks, embeddings, model):
    question_embedding = model.encode(question)

    scores = []

    for chunk_embedding in embeddings:
        dot_product = np.dot(question_embedding, chunk_embedding)

        magnitude = (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        similarity = dot_product / magnitude if magnitude != 0 else 0
        scores.append(similarity)

    top_indices = np.argsort(scores)[::-1][:3]

    for index in top_indices:
        print(f"\nSimilarity: {scores[index]:.4f}")
        print(chunks[index])
        print("-" * 50)


def main():
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)

    if not chunks or not text.strip():
        print("No extractable text found in the PDF.")
        return

    print("Total chunks:", len(chunks))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    question = input("\nAsk a question: ")

    search_chunks(question, chunks, embeddings, model)


if __name__ == "__main__":
    main()
