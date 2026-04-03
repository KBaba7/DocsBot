from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import User
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStoreService
from app.services.web_search import build_web_search_tool


class VectorSearchInput(BaseModel):
    query: str = Field(..., description="The user question to answer from uploaded documents.")


LANGGRAPH_CHECKPOINTER = MemorySaver()


def _route_tools(state: MessagesState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"


def build_agent(*, db: Session, user: User):
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for agent responses.")
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.model_name, temperature=0)
    document_service = DocumentService()
    vector_store = VectorStoreService()
    web_search_tool = build_web_search_tool()

    def vector_search(query: str) -> str:
        resolved_hashes = document_service.resolve_relevant_document_hashes(db, user=user, query=query)
        if not resolved_hashes:
            return "No uploaded documents are available for this user."
        matches = vector_store.similarity_search(db=db, query=query, file_hashes=resolved_hashes, k=settings.retrieval_k)
        if not matches:
            return f"No vector matches found for hashes: {resolved_hashes}"
        lines = ["Vector evidence (cite document + page + excerpt in final answer):"]
        for index, match in enumerate(matches, start=1):
            page_number = match["metadata"].get("page_number")
            page_label = str(page_number) if page_number is not None else "unknown"
            document_id = match["metadata"].get("document_id")
            score_parts = [f"distance={match['distance']:.4f}"]
            if "rerank_score" in match:
                score_parts.append(f"rerank_score={match['rerank_score']:.4f}")
            lines.append(f"{index}. document_id={document_id} | document={match['metadata']['filename']} | page={page_label} | {' | '.join(score_parts)}")
            lines.append(f"   excerpt: {match['content'][:900].replace(chr(10), ' ')}")
        return "\n\n".join(lines)

    vector_tool = StructuredTool.from_function(
        func=vector_search,
        name="vector_search",
        description=(
            "Searches the current user's uploaded documents. "
            "The tool automatically resolves the most relevant documents for the current user before chunk retrieval."
        ),
        args_schema=VectorSearchInput,
    )

    tools = [vector_tool]
    prompt = (
        "You are a document QA agent. Prefer vector_search for questions about the user's uploaded documents. "
        "Do NOT include any 'Sources' section, citation list, footnotes, chunk ids, or hashes in the final answer text. "
        "Only provide the concise user-facing answer. "
        "Citation metadata is handled separately by the application. "
        "Do not claim evidence that is not present in tool outputs."
    )
    if web_search_tool is not None:
        tools.append(web_search_tool)
        prompt += " Use web search only when the answer depends on external or current information."
    else:
        prompt += " Web search is currently unavailable in this environment."

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    system_prompt = SystemMessage(content=prompt)

    def agent_node(state: MessagesState):
        response = llm_with_tools.invoke([system_prompt, *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_tools, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=LANGGRAPH_CHECKPOINTER)
