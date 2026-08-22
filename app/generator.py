import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


# Create Hugging Face client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)


def generate_answer(prompt):

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:groq",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    prompt = "Explain Generative AI in one simple sentence."

    answer = generate_answer(prompt)

    print("AI:", answer)