from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext
from langchain_community.tools import DuckDuckGoSearchResults

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown



load_dotenv()

model = 'google-gla:gemini-2.0-flash'

@dataclass
class Deps:
    client: AsyncClient


web_search_agent = Agent(
    model,
    system_prompt=f'You are an expert at researching the web to answer user questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
    deps_type=Deps,
    retries=2
)






@web_search_agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """

    search = DuckDuckGoSearchResults(output_format="list",max_results=5)
    web_results = search.invoke(web_query)

    results = []
    for item in web_results[:5]:
        #print (item)
        title = item.get('title', '')
        description = item.get('snippet', '')
        url = item.get('link', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."

async def main():
    async with AsyncClient() as client:
        deps = Deps(client=client)

        result = await web_search_agent.run(
            'Give me some articles talking about the new release of React 19.', deps=deps
        )
        
        debug(result)
        print('Response:', result.data)

        console = Console()
        console.log(Markdown(result.data))

if __name__ == '__main__':
    asyncio.run(main())