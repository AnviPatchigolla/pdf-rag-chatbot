import pymupdf
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


def main():
    text = extract_text("data/sample.pdf")
    chunks = chunk_text(text)

    print("Total chunks:", len(chunks))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    print("Number of embeddings:", len(embeddings))
    print("Dimensions:", len(embeddings[0]))
    print("First 10 values:", embeddings[0][:10])


if __name__ == "__main__":
    main()
