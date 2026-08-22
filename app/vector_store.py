from pathlib import Path

from langchain_community.vectorstores import FAISS

from chunker import create_chunks
from embeddings import create_embedding_model
from loader import load_pdf


VECTOR_STORE_PATH = Path("vectorstore")


def create_vector_store():

    documents = load_pdf()

    chunks = create_chunks(documents)

    embeddings = create_embedding_model()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTOR_STORE_PATH.mkdir(exist_ok=True)

    vector_store.save_local(
        str(VECTOR_STORE_PATH)
    )

    print("\nVector store created successfully!")
    print(f"Total chunks stored: {len(chunks)}")


if __name__ == "__main__":
    create_vector_store()