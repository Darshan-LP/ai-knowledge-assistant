from pathlib import Path

from langchain_community.vectorstores import FAISS

from embeddings import create_embedding_model


VECTOR_STORE_PATH = Path("vectorstore")


def load_vector_store():

    embeddings = create_embedding_model()

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def retrieve_documents(question, k=3):

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=k
    )

    return documents


if __name__ == "__main__":

    question = input("Enter your question: ")

    documents = retrieve_documents(question)

    print("\nRetrieved Documents:")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):

        print(f"\nResult {i}")
        print("-" * 60)

        print(document.page_content)