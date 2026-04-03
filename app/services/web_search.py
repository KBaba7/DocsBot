import json

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings


class WebSearchInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str = Field(default="", description="Search query for external/current information.")
    cursor: int | None = Field(default=None, description="Optional cursor from model/tool planning. Ignored.")
    id: str | int | None = Field(default=None, description="Optional id from model/tool planning. Ignored.")


def _build_tavily_tool(api_key: str):
    try:
        from tavily import TavilyClient
    except Exception:
        return None

    client = TavilyClient(api_key=api_key)

    def tavily_search(query: str = "", cursor: int | None = None, id: str | int | None = None, **kwargs) -> str:
        query_text = query.strip()
        _ = (cursor, id, kwargs)
        if not query_text:
            return "Web search was requested without a query."
        try:
            result = client.search(query=query_text, search_depth="advanced")
        except Exception as exc:
            return f"Tavily search failed: {exc}"
        rows = result.get("results", []) if isinstance(result, dict) else []
        if not rows:
            return "No web results found."

        lines = ["Web search results (cite website URLs used):"]
        for index, row in enumerate(rows[:5], start=1):
            title = str(row.get("title") or "Untitled result")
            url = str(row.get("url") or "").strip()
            snippet = str(row.get("content") or "").strip().replace("\n", " ")
            snippet = snippet[:500]
            lines.append(f"{index}. title: {title}")
            lines.append(f"   url: {url or 'N/A'}")
            if snippet:
                lines.append(f"   snippet: {snippet}")
        lines.append("\nRaw response:")
        lines.append(json.dumps(result, ensure_ascii=True))
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=tavily_search,
        name="web_search",
        description="Search the web for current/external information using Tavily.",
        args_schema=WebSearchInput,
    )


def build_web_search_tool():
    settings = get_settings()
    provider = settings.web_search_provider.lower()

    if provider == "tavily":
        if not settings.tavily_api_key:
            return None
        return _build_tavily_tool(settings.tavily_api_key)

    try:
        return DuckDuckGoSearchResults(num_results=5)
    except Exception:
        return None
