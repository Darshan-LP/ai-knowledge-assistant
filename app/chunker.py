from loader import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=75,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Total chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    documents = load_pdf()

    chunks = create_chunks(documents)

    print("\n--- Chunks ---\n")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}")
        print("-" * 50)
        print("Chunk ID:", chunk.metadata.get("chunk_id"))
        print("Source:", chunk.metadata.get("source"))
        print("Page:", chunk.metadata.get("page_label"))
        print()
        print(chunk.page_content)