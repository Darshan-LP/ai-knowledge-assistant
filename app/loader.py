from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PDF_PATH = Path("data/documents")


def load_pdf():
    pdf_files = list(PDF_PATH.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF found inside data/documents")

    pdf_file = pdf_files[0]

    print(f"Loading PDF: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    print(f"Total pages loaded: {len(documents)}")

    return documents


if __name__ == "__main__":
    documents = load_pdf()

    print("\n--- First page content ---\n")
    print(documents[0].page_content)