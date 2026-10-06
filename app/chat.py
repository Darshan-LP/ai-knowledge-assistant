from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.graph import agent_graph
from app.auth.dependencies import get_current_user
from app.auth.schemas import ChatRequest
from app.database import get_db
from app.models import Conversation, Message

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post("")
def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = int(current_user["user_id"])

    if request.conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == request.conversation_id,
                Conversation.user_id == user_id
            )
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied."
            )

    else:
        conversation = Conversation(
            user_id=user_id,
            title=request.message[:50]
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message
    )

    db.add(user_message)
    db.commit()

    result = agent_graph.invoke(
        {
            "question": request.message,
            "answer": "",
            "documents": [],
            "route": ""
        }
    )

    answer = result.get("answer", "")

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer
    )

    db.add(assistant_message)
    db.commit()

    sources = []
    for document in result.get("documents", []):
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page_label", "Unknown")
        chunk_id = document.metadata.get("chunk_id", "Unknown")

        sources.append(
            {
                "source": source.replace("\\", "/").split("/")[-1],
                "page": page,
                "chunk_id": chunk_id
            }
        )

    return {
        "conversation_id": conversation.id,
        "route": result.get("route", "Unknown"),
        "answer": answer,
        "sources": sources
    }


@router.get("/conversations")
def get_user_conversations(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = int(current_user["user_id"])

    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )

    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at
        }
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}")
def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = int(current_user["user_id"])

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied."
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at
            }
            for m in messages
        ]
    }