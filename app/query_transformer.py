from llm import create_llm_client


def transform_query(question):

    client = create_llm_client()

    prompt = f"""
You are a search query optimizer for a document-based RAG system.

Rewrite the user's question into a concise search query
that will help retrieve the most relevant information
from a company policy document.

Rules:
1. Preserve the original meaning.
2. Do not answer the question.
3. Do not add information that is not present in the question.
4. Keep important keywords.
5. Return ONLY the rewritten search query.
6. Do not use quotes or explanations.

User Question:
{question}

Search Query:
"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":

    question = input("Enter your question: ")

    transformed_query = transform_query(question)

    print("\nOriginal Question:")
    print(question)

    print("\nTransformed Query:")
    print(transformed_query)