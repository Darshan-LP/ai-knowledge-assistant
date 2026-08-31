from hybrid_retriever import hybrid_retrieve
from query_transformer import transform_query
from reranker import rerank_documents
from llm import create_llm_client


FALLBACK_ANSWER = (
    "I couldn't find the answer in the provided document."
)


def build_context(documents):

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        content = document.page_content

        context_parts.append(
            f"""
SOURCE [{i}]
Document: {source}
Page: {page}

Content:
{content}
"""
        )

    return "\n".join(context_parts)


def generate_rag_answer(question):

    # --------------------------------------------------
    # Step 1: Transform user question
    # --------------------------------------------------

    search_query = transform_query(question)

    print(f"\nOriginal Question: {question}")
    print(f"Search Query: {search_query}")


   # --------------------------------------------------
    # Step 2: Hybrid Retrieval
    # --------------------------------------------------

    hybrid_results = hybrid_retrieve(
        search_query
    )


    # --------------------------------------------------
    # Step 2.1: Stop if no documents found
    # --------------------------------------------------

    if not hybrid_results:

        return FALLBACK_ANSWER, []


    # --------------------------------------------------
    # Step 2.2: Extract documents
    # --------------------------------------------------

    documents = [
        result["document"]
        for result in hybrid_results
    ]


    # --------------------------------------------------
    # Step 2.3: Rerank documents
    # --------------------------------------------------

    reranked_results = rerank_documents(
        question,
        documents,
        top_k=2
    )


    # --------------------------------------------------
    # Step 2.4: Extract reranked documents
    # --------------------------------------------------

    documents = [
        document
        for document, score in reranked_results
    ]


    # --------------------------------------------------
    # Step 2.5: Stop if reranking returns nothing
    # --------------------------------------------------

    if not documents:

        return FALLBACK_ANSWER, []


    # --------------------------------------------------
    # Step 3: Build context
    # --------------------------------------------------

    context = build_context(
        documents
    )


    # --------------------------------------------------
    # Step 4: Create strict RAG prompt
    # --------------------------------------------------

    prompt = f"""
You are a document-based AI assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided sources.

IMPORTANT RULES:

1. Use ONLY the provided sources to answer the question.
2. Do NOT use your general knowledge.
3. Do NOT make assumptions or guesses.
4. Do NOT invent or create information that is not present
   in the sources.
5. If the sources do not contain enough information to answer
   the question, respond exactly with:
   "{FALLBACK_ANSWER}"
6. Keep the answer concise and direct.
7. When an answer is supported by a source, include its
   source number using [1], [2], etc.
8. Only cite sources that actually support your answer.

Retrieved Sources:
--------------------
{context}
--------------------

User Question:
{question}

Answer:
"""


    # --------------------------------------------------
    # Step 5: Send prompt to LLM
    # --------------------------------------------------

    import time

    client = create_llm_client()

    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    elapsed = time.time() - start_time

    print(
        f"\nModel used: {response.model}"
    )

    print(
        f"Response time: {elapsed:.2f} seconds"
    )


    # --------------------------------------------------
    # Step 6: Extract answer
    # --------------------------------------------------

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


    return answer, documents


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )


    answer, documents = generate_rag_answer(
        question
    )


    print("\nQuestion:")
    print(question)


    print("\nRAG Answer:")
    print("=" * 60)
    print(answer)


    print("\nSources:")
    print("=" * 60)


    if documents:

        for i, document in enumerate(
            documents,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page_label",
                "Unknown"
            )

            source_name = source.replace(
                "\\",
                "/"
            ).split("/")[-1]

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown"
            )

            print(
                f"[{i}] {source_name} "
                f"— Page {page} "
                f"— Chunk {chunk_id}"
            )

    else:

        print("No sources found.")