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


def retrieve_documents(question, k=5, threshold=1.0):

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    filtered_documents = []

    for document, score in results:

        if score <= threshold:
            filtered_documents.append(document)

    return filtered_documents


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = retrieve_documents(
        question,
        k=5,
        score_threshold=1.0
    )

    print("\nRetrieved Documents After Threshold:")
    print("=" * 60)

    if not results:

        print("No sufficiently relevant documents found.")

    else:

        for i, (document, score) in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 60)

            print(f"Distance Score: {score}")

            print("\nContent:")
            print(document.page_content)

            print("\nMetadata:")
            print(document.metadata)