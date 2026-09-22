from app.hybrid_retriever import hybrid_retrieve
from app.query_transformer import transform_query
from app.reranker import rerank_documents
from app.llm import create_llm_client


FALLBACK_ANSWER = (
    "I couldn't find the answer in the provided document."
)


# ======================================================
# BUILD CONTEXT
# ======================================================

def build_context(documents):

    context_parts = []

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


# ======================================================
# GROUNDING VALIDATION
# ======================================================

def validate_grounding(
    question,
    answer,
    context
):

    client = create_llm_client()

    validation_prompt = f"""
You are a grounding validator for a document-based AI assistant.

Your job is to determine whether the proposed answer
is fully supported by the provided sources.

Rules:

1. Use ONLY the provided sources.
2. Do NOT use outside knowledge.
3. The answer must be supported by the source content.
4. If the answer contains information that is not supported
   by the sources, mark it as NOT GROUNDED.
5. If the answer is supported by the sources, mark it as GROUNDED.
6. Respond with ONLY one word:

GROUNDED

or

NOT GROUNDED


Retrieved Sources:
--------------------
{context}
--------------------

User Question:
{question}

Proposed Answer:
{answer}

Validation:
"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": validation_prompt
            }
        ]
    )

    validation_result = (
        response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    print(
        f"Grounding validation: {validation_result}"
    )

    return validation_result == "GROUNDED"


# ======================================================
# GENERATE RAG ANSWER
# ======================================================

def generate_rag_answer(question):

    # --------------------------------------------------
    # Step 1: Transform user question
    # --------------------------------------------------

    search_query = transform_query(
        question
    )

    print(
        f"\nOriginal Question: {question}"
    )

    print(
        f"Search Query: {search_query}"
    )


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

# --------------------------------------------------
# FALLBACK ANSWER
# --------------------------------------------------

    if answer == FALLBACK_ANSWER:

        print(
            "Fallback answer detected. "
            "Skipping grounding validation."
        )

        return answer, documents


    # --------------------------------------------------
    # GROUNDING VALIDATION
    # --------------------------------------------------

    is_grounded = validate_grounding(
        question,
        answer,
        context
    )


    # --------------------------------------------------
    # Step 8: Reject unsupported answer
    # --------------------------------------------------

    if not is_grounded:

        print(
            "Answer rejected because it was not grounded."
        )

        return FALLBACK_ANSWER, documents


    # --------------------------------------------------
    # Step 9: Return validated answer
    # --------------------------------------------------

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