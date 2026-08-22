from loader import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    documents = load_pdf()

    chunks = create_chunks(documents)

    print("\n--- Chunks ---\n")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}")
        print("-" * 50)
        print(chunk.page_content)