from sentence_transformers import CrossEncoder


# ======================================================
# CONFIGURATION
# ======================================================

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RERANK_TOP_K = 2


# ======================================================
# LOAD RERANKER MODEL
# ======================================================

_reranker = None


def load_reranker():

    global _reranker

    if _reranker is None:

        print("Loading reranker model...")

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

    return _reranker


# ======================================================
# RERANK DOCUMENTS
# ======================================================

def rerank_documents(
    question,
    documents,
    top_k=RERANK_TOP_K
):

    if not documents:

        return []


    reranker = load_reranker()


    # ----------------------------------------------
    # Create question-document pairs
    # ----------------------------------------------

    pairs = []

    for document in documents:

        pairs.append(
            (
                question,
                document.page_content
            )
        )


    # ----------------------------------------------
    # Get relevance scores
    # ----------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # ----------------------------------------------
    # Combine documents with scores
    # ----------------------------------------------

    ranked_results = []

    for document, score in zip(
        documents,
        scores
    ):

        ranked_results.append(
            (
                document,
                float(score)
            )
        )


    # ----------------------------------------------
    # Sort highest relevance first
    # ----------------------------------------------

    ranked_results.sort(
        key=lambda item: item[1],
        reverse=True
    )


    # ----------------------------------------------
    # Return only top documents
    # ----------------------------------------------

    return ranked_results[:top_k]


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    from hybrid_retriever import hybrid_retrieve


    question = input(
        "Enter your question: "
    )


    hybrid_results = hybrid_retrieve(
        question
    )


    documents = [
        result["document"]
        for result in hybrid_results
    ]


    reranked_results = rerank_documents(
        question,
        documents
    )


    print("\n")
    print("=" * 70)
    print("RERANKING RESULTS")
    print("=" * 70)


    if not reranked_results:

        print(
            "\nNo documents available for reranking."
        )


    else:

        for i, (
            document,
            score
        ) in enumerate(
            reranked_results,
            start=1
        ):

            print(
                f"\nRank {i}"
            )

            print("-" * 60)

            print(
                f"Reranker Score: "
                f"{score:.6f}"
            )

            print(
                f"Chunk ID: "
                f"{document.metadata.get('chunk_id')}"
            )

            print(
                f"Page: "
                f"{document.metadata.get('page_label')}"
            )

            print("\nContent:")

            print(
                document.page_content
            )