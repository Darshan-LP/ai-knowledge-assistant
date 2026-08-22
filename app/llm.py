import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def create_llm_client():

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN")
    )

    return client


def generate_answer(question):

    client = create_llm_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:groq",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    question = "What is remote work?"

    answer = generate_answer(question)

    print("\nQuestion:")
    print(question)

    print("\nAI:")
    print(answer)