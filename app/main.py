from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import generate_rag_answer


app = FastAPI(
    title="AI Knowledge Assistant API",
    version="1.0.0"
)


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def home():

    return {
        "message": "AI Knowledge Assistant API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer, documents = generate_rag_answer(
        request.question
    )

    sources = []

    for document in documents:

        sources.append(
            {
                "source": document.metadata.get(
                    "source",
                    "Unknown"
                ).split("\\")[-1],

                "page": document.metadata.get(
                    "page_label",
                    "Unknown"
                ),

                "chunk_id": document.metadata.get(
                    "chunk_id",
                    "Unknown"
                )
            }
        )

    return {
        "answer": answer,
        "sources": sources
    }