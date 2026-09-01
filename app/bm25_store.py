from pathlib import Path
import pickle

from rank_bm25 import BM25Okapi

from app.chunker import create_chunks
from app.loader import load_pdf


BM25_STORE_PATH = Path("bm25_store")


def tokenize(text):

    return text.lower().split()


def create_bm25_store():

    documents = load_pdf()

    chunks = create_chunks(documents)

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    BM25_STORE_PATH.mkdir(exist_ok=True)

    with open(
        BM25_STORE_PATH / "bm25.pkl",
        "wb"
    ) as file:

        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks
            },
            file
        )

    print("\nBM25 index created successfully!")
    print(f"Total chunks indexed: {len(chunks)}")

    return bm25, chunks


def load_bm25_index():

    index_path = BM25_STORE_PATH / "bm25.pkl"

    if not index_path.exists():

        raise FileNotFoundError(
            "BM25 index not found. "
            "Run 'python app\\bm25_store.py' first."
        )

    with open(
        index_path,
        "rb"
    ) as file:

        data = pickle.load(file)

    return data["bm25"], data["chunks"]


def search_bm25(
    question,
    bm25,
    chunks,
    k=3
):

    tokenized_question = tokenize(question)

    scores = bm25.get_scores(
        tokenized_question
    )

    ranked_results = sorted(
        zip(chunks, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results[:k]


if __name__ == "__main__":

    create_bm25_store()

    print("\nBM25 index saved successfully!")