from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.graph import agent_graph


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

    result = agent_graph.invoke(
        {
            "question": request.question,
            "answer": "",
            "documents": [],
            "route": ""
        }
    )

    sources = []

    for document in result.get(
        "documents",
        []
    ):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "Unknown"
        )

        sources.append(
            {
                "source": source.replace(
                    "\\",
                    "/"
                ).split("/")[-1],

                "page": page,

                "chunk_id": chunk_id
            }
        )

    return {
        "route": result.get(
            "route",
            "Unknown"
        ),

        "answer": result.get(
            "answer",
            ""
        ),

        "sources": sources
    }