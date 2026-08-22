from retriever import retrieve_documents
from llm import create_llm_client


def build_context(documents):

    context = ""

    for document in documents:
        context += document.page_content
        context += "\n\n"

    return context


def generate_rag_answer(question):

    # Step 1: Retrieve relevant documents
    documents = retrieve_documents(
        question,
        k=2
    )

    # Step 2: Build context
    context = build_context(documents)

    # Step 3: Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say: "I couldn't find the answer in the provided document."

Do not use outside knowledge.

Context:
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

    for document in documents:
        print(document.metadata)