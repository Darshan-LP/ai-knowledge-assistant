from retriever import retrieve_documents
from llm import create_llm_client


def build_context(documents):

    context_parts = []

    for i, (document, score) in enumerate(documents, start=1):

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
SOURCE {i}
Document: {source}
Page: {page}
Relevance Score: {score}

Content:
{content}
"""
        )

    return "\n".join(context_parts)


def generate_rag_answer(question):

    # Step 1: Retrieve relevant documents
    documents = retrieve_documents(
        question,
        k=5
    )

    # Step 2: Build context
    context = build_context(documents)

    # Step 3: Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the sources below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the sources, say:
   "I couldn't find the answer in the provided document."
3. Do not invent facts.
4. Give a concise and direct answer.

Sources:
--------------------
{context}
--------------------

User Question:
{question}

Answer:
"""

    # Step 4: Send prompt to LLM
    client = create_llm_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:groq",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return answer, documents


if __name__ == "__main__":

    question = "How many sick leave days are provided?"

    answer, documents = generate_rag_answer(question)

    print("\nQuestion:")
    print(question)

    print("\nRAG Answer:")
    print("=" * 60)
    print(answer)

    print("\nSources:")
    print("=" * 60)

    for document, score in documents:
        print("Document:", document.metadata.get("source", "Unknown"))
        print("Page:", document.metadata.get("page_label", "Unknown"))
        print("Score:", score)
        print()