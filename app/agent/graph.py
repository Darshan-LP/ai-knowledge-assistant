from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.rag_pipeline import generate_rag_answer


class AgentState(TypedDict):
    question: str
    answer: str
    documents: list
    route: str


def rag_node(state: AgentState):

    question = state["question"]

    answer, documents = generate_rag_answer(
        question
    )

    return {
        "answer": answer,
        "documents": documents
    }


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node(
        "rag",
        rag_node
    )

    graph.add_edge(
        START,
        "rag"
    )

    graph.add_edge(
        "rag",
        END
    )

    return graph.compile()


agent_graph = build_graph()


if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    result = agent_graph.invoke(
        {
            "question": question,
            "answer": "",
            "documents": [],
            "route": ""
        }
    )

    print("\nAnswer:")
    print("=" * 60)
    print(result["answer"])

    print("\nSources:")
    print("=" * 60)

    for i, document in enumerate(
        result["documents"],
        start=1
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

        source_name = source.replace(
            "\\",
            "/"
        ).split("/")[-1]

        print(
            f"[{i}] {source_name} "
            f"— Page {page} "
            f"— Chunk {chunk_id}"
        )