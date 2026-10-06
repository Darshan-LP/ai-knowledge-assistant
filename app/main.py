from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.agent.graph import agent_graph
from app.auth.dependencies import get_current_user
from app.auth.routes import router as auth_router
from app.chat import router as chat_router


app = FastAPI(
    title="AI Knowledge Assistant API",
    version="1.0.0"
)


app.include_router(auth_router)
app.include_router(chat_router)


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def home():

    return {
        "message": "AI Knowledge Assistant API is running"
    }


@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    current_user: dict = Depends(get_current_user)
):
    result = agent_graph.invoke(
        {
            "question": request.question,
            "answer": "",
            "documents": [],
            "route": ""
        }
    )

    sources = []

    for document in result.get("documents", []):
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
                "source": source.replace("\\", "/").split("/")[-1],
                "page": page,
                "chunk_id": chunk_id
            }
        )

    return {
        "user": current_user["username"],
        "route": result.get("route", "Unknown"),
        "answer": result.get("answer", ""),
        "sources": sources
    }