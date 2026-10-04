from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.rag_pipeline import generate_rag_answer


class AgentState(TypedDict):

    question: str
    answer: str
    documents: list
    route: str


# ======================================================
# ROUTER NODE
# ======================================================

def router_node(state: AgentState):

    question = state["question"].lower().strip()

    # --------------------------------------------------
    # DIRECT ROUTE
    # --------------------------------------------------

    if question in ["hi", "hello", "hey"]:

        route = "direct"

    # --------------------------------------------------
    # WEATHER TOOL
    # --------------------------------------------------

    elif "weather" in question:

        route = "tool"

    # --------------------------------------------------
    # CALCULATOR TOOL
    # --------------------------------------------------

    elif (
        "calculate" in question
        or "calculator" in question
        or "employee" in question
    ):

        route = "tool"

    # --------------------------------------------------
    # RAG ROUTE
    # --------------------------------------------------

    else:

        route = "rag"

    print(
        f"Route selected: {route}"
    )

    return {
        "route": route
    }


# ======================================================
# RAG NODE
# ======================================================

def rag_node(state: AgentState):

    question = state["question"]

    answer, documents = generate_rag_answer(
        question
    )

    return {
        "answer": answer,
        "documents": documents
    }


# ======================================================
# DIRECT NODE
# ======================================================

def direct_node(state: AgentState):

    return {
        "answer": "Hello! How can I help you?"
    }


# ======================================================
# TOOL NODE
# ======================================================

# ======================================================
# TOOL NODE
# ======================================================


def tool_node(state: AgentState):

    from app.tools import execute_tool

    question = state["question"]

    result = execute_tool(question)

    # TOOL ERROR
    if "error" in result:
        return {
            "answer": result["error"]
        }

    # WEATHER TOOL
    if result.get("tool") == "weather":
        answer = (
            f"Location: {result['location']}\n"
            f"Temperature: {result['temperature']}\n"
            f"Condition: {result['condition']}"
        )

        return {
            "answer": answer
        }

    # CALCULATOR TOOL
    if result.get("tool") == "calculator":
        answer = (
            f"Expression: {result['expression']}\n"
            f"Result: {result['result']}"
        )

        return {
            "answer": answer
        }

    # DATABASE TOOL
    if result.get("tool") == "database":
        employee = result["data"]

        answer = (
            f"Employee ID: {employee['employee_id']}\n"
            f"Name: {employee['name']}\n"
            f"Department: {employee['department']}\n"
            f"Designation: {employee['designation']}"
        )

        return {
            "answer": answer
        }

    # UNKNOWN TOOL
    return {
        "answer": "Unsupported tool result."
    }


# ======================================================
# BUILD LANGGRAPH
# ======================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    # --------------------------------------------------
    # Add nodes
    # --------------------------------------------------

    graph.add_node(
        "router",
        router_node
    )

    graph.add_node(
        "rag",
        rag_node
    )

    graph.add_node(
        "direct",
        direct_node
    )

    graph.add_node(
        "tool",
        tool_node
    )

    # --------------------------------------------------
    # START → ROUTER
    # --------------------------------------------------

    graph.add_edge(
        START,
        "router"
    )

    # --------------------------------------------------
    # ROUTER → RAG / TOOL / DIRECT
    # --------------------------------------------------

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "rag",
            "tool": "tool",
            "direct": "direct"
        }
    )

    # --------------------------------------------------
    # RAG → END
    # --------------------------------------------------

    graph.add_edge(
        "rag",
        END
    )

    # --------------------------------------------------
    # TOOL → END
    # --------------------------------------------------

    graph.add_edge(
        "tool",
        END
    )

    # --------------------------------------------------
    # DIRECT → END
    # --------------------------------------------------

    graph.add_edge(
        "direct",
        END
    )

    return graph.compile()


# ======================================================
# CREATE GRAPH
# ======================================================

agent_graph = build_graph()


# ======================================================
# TEST
# ======================================================

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

    print(
        result["answer"]
    )

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