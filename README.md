# AI Knowledge Assistant (v1.0.0) 🚀

A production-ready Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **LangChain**, **FAISS**, and **JWT Authentication**. Includes fully containerized deployment via **Docker**, an automated **GitHub Actions CI/CD** pipeline, and an end-to-end **RAG Evaluation Suite**.

---

## 🌟 Architecture & Key Features

* **RAG Pipeline:** Leverages LangChain & FAISS vector store for semantic context retrieval and LLM answer generation.
* **JWT Security:** Standard OAuth2 Bearer token authentication securing API routes (`/chat`).
* **FastAPI Backend:** High-performance async Python backend with auto-generated OpenAPI / Swagger UI.
* **Containerization:** Optimized Docker image built on `python:3.12-slim`.
* **CI/CD Pipeline:** Automated unit tests and Docker build verification via GitHub Actions.
* **Evaluation Framework:** Built-in benchmarking suite assessing response latency, endpoint status, and semantic keyword relevance.

---

## 📁 Repository Structure

```text
ai-knowledge-assistant/
├── app/
│   ├── main.py              # FastAPI entry point & routers
│   ├── retriever.py         # Vector database & RAG pipeline
│   └── services/            # Core business logic & auth services
├── eval/
│   └── evaluate_rag.py      # RAG performance & relevance evaluation
├── tests/
│   └── test_chat_auth.py    # Pytest unit & integration tests
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── Dockerfile               # Multi-stage optimized Docker build
├── .dockerignore            # Build context filter
├── requirements.txt         # Dependency lockfile
├── eval_results.json        # Evaluation output logs
└── README.md