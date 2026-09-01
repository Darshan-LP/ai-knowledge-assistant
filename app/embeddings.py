from langchain_huggingface import HuggingFaceEmbeddings

from app.chunker import create_chunks
from app.loader import load_pdf


MODEL_NAME = "models/all-MiniLM-L6-v2"


def create_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    return embeddings


if __name__ == "__main__":

    documents = load_pdf()

    chunks = create_chunks(documents)

    embeddings = create_embedding_model()

    print(f"\nNumber of chunks: {len(chunks)}")

    vector = embeddings.embed_query(
        "How many PTO days can employees carry over?"
    )

    print(f"Embedding vector size: {len(vector)}")

    print("\nFirst 10 numbers:")
    print(vector[:10])