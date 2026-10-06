from langgraph.graph import StateGraph, START, END

from .state import ExtractionState

from .nodes import (
    ingest_node,
    clean_node,
    chunk_node,
    vectorstore_node,
    retrieval_node,
    extraction_node,
    validate_node,
    output_node,
    route_after_ingest,
    route_after_validate,
)


def create_extraction_graph():
    """ingest -> clean -> chunk -> vectorstore -> retrieve -> extract -> validate -> output

    * ingest failure / empty file   -> END (error in state, no crash)
    * validate finds a failed LLM section -> back to retrieve (wider k), max N attempts
    """
    graph = StateGraph(ExtractionState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("clean", clean_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("vectorstore", vectorstore_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("extract", extraction_node)
    graph.add_node("validate", validate_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", route_after_ingest, {"clean": "clean", "end": END})
    graph.add_edge("clean", "chunk")
    graph.add_edge("chunk", "vectorstore")
    graph.add_edge("vectorstore", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges("validate", route_after_validate,
                                {"retry": "retrieve", "output": "output"})
    graph.add_edge("output", END)

    return graph.compile()
