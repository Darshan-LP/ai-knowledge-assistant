from app.retriever import retrieve_documents
from app.bm25_store import load_bm25_index, search_bm25


# ======================================================
# CONFIGURATION
# ======================================================

FAISS_K = 3
BM25_K = 3

HYBRID_K = 3

FAISS_WEIGHT = 0.70
BM25_WEIGHT = 0.30

HYBRID_THRESHOLD = 0.30


# ======================================================
# NORMALIZE BM25 SCORES
# ======================================================

def normalize_bm25_scores(results):

    if not results:
        return []

    scores = [
        score
        for _, score in results
    ]

    max_score = max(scores)

    if max_score == 0:

        return [
            (document, 0)
            for document, _ in results
        ]

    normalized_results = []

    for document, score in results:

        normalized_score = score / max_score

        normalized_results.append(
            (
                document,
                normalized_score
            )
        )

    return normalized_results


# ======================================================
# NORMALIZE FAISS SCORES
# ======================================================

def normalize_faiss_scores(results):

    if not results:
        return []

    distances = [
        distance
        for _, distance in results
    ]

    min_distance = min(distances)
    max_distance = max(distances)

    if max_distance == min_distance:

        return [
            (document, 1.0)
            for document, _ in results
        ]

    normalized_results = []

    for document, distance in results:

        relevance_score = (
            (max_distance - distance)
            /
            (max_distance - min_distance)
        )

        normalized_results.append(
            (
                document,
                relevance_score
            )
        )

    return normalized_results


# ======================================================
# HYBRID RETRIEVAL
# ======================================================

def hybrid_retrieve(question):

    # --------------------------------------------------
    # 1. FAISS Retrieval
    # --------------------------------------------------

    faiss_results = retrieve_documents(
        question,
        k=FAISS_K,
        threshold=1.5
    )

    # --------------------------------------------------
    # 2. Load BM25 Index
    # --------------------------------------------------

    bm25, chunks = load_bm25_index()

    # --------------------------------------------------
    # 3. BM25 Retrieval
    # --------------------------------------------------

    bm25_results = search_bm25(
        question,
        bm25,
        chunks,
        k=BM25_K
    )

    # --------------------------------------------------
    # 4. Normalize Scores
    # --------------------------------------------------

    normalized_faiss = normalize_faiss_scores(
        faiss_results
    )

    normalized_bm25 = normalize_bm25_scores(
        bm25_results
    )

    # --------------------------------------------------
    # 5. Combine Results
    # --------------------------------------------------

    combined_results = {}

    # Add FAISS Results

    for document, score in normalized_faiss:

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id is None:
            continue

        combined_results.setdefault(
            chunk_id,
            {
                "document": document,
                "faiss_score": 0,
                "bm25_score": 0
            }
        )

        combined_results[
            chunk_id
        ]["faiss_score"] = score


    # Add BM25 Results

    for document, score in normalized_bm25:

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id is None:
            continue

        combined_results.setdefault(
            chunk_id,
            {
                "document": document,
                "faiss_score": 0,
                "bm25_score": 0
            }
        )

        combined_results[
            chunk_id
        ]["bm25_score"] = score


    # --------------------------------------------------
    # 6. Calculate Hybrid Score
    # --------------------------------------------------

    for result in combined_results.values():

        hybrid_score = (
            FAISS_WEIGHT
            * result["faiss_score"]
        ) + (
            BM25_WEIGHT
            * result["bm25_score"]
        )

        result["hybrid_score"] = hybrid_score


    # --------------------------------------------------
    # 7. Sort Results
    # --------------------------------------------------

    ranked_results = sorted(
        combined_results.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True
    )


    # --------------------------------------------------
    # 8. Filter Low-Relevance Results
    # --------------------------------------------------

    filtered_results = [
        result
        for result in ranked_results
        if result["hybrid_score"] >= HYBRID_THRESHOLD
    ]


    # --------------------------------------------------
    # 9. Return Top Results
    # --------------------------------------------------

    return filtered_results[:HYBRID_K]


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    results = hybrid_retrieve(
        question
    )

    print("\n")
    print("=" * 70)
    print("WEIGHTED HYBRID RETRIEVAL RESULTS")
    print("=" * 70)

    if not results:

        print(
            "\nNo relevant documents found."
        )

    else:

        for i, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            print(
                f"\nHybrid Result {i}"
            )

            print("-" * 60)

            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.6f}"
            )

            print(
                f"FAISS Score: "
                f"{result['faiss_score']:.6f}"
            )

            print(
                f"BM25 Score: "
                f"{result['bm25_score']:.6f}"
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