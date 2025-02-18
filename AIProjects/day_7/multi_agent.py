from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from agent_prompts import SYSTEM_PROMPTS
from pydantic_ai.messages import ToolReturnPart

from agent_schema import ResearchQuery, ResearchOutput, ResearchResult, CodeExecutionRequest, CodeExecutionResult
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL


supervisor_prompt = SYSTEM_PROMPTS["supervisor"]
researcher_prompt = SYSTEM_PROMPTS["researcher"]
coding_prompt = SYSTEM_PROMPTS["coding"]
report_writer_prompt = SYSTEM_PROMPTS["report_writer"]

print(supervisor_prompt) 


#supervisor_prompt = "You are an ai agent. Use `researcher_tool` to search the web to answer questions, and `coding_tool` to answer questions that involve math. Always formulate a plan how to best use the tools to answer the question.  Then execute that plan."
#researcher_prompt = "You are an ai agent. Use `search_web` to search the web to answer questions."
#coding_prompt = "You are an ai agent who answers questions by generating python code then using the `python_code_execution` tool to execute your python to obtain the answer.  If this tool gives an error, modify the python code accordingly and try again."


supervisor_agent = Agent(  
    'google-gla:gemini-2.0-flash',
    system_prompt=supervisor_prompt,
)

researcher_agent = Agent(  
    'google-gla:gemini-2.0-flash',
    system_prompt=researcher_prompt,
    deps_type=ResearchQuery,
    result_type=ResearchOutput,
)

coding_agent = Agent(  
    'google-gla:gemini-2.0-flash',
    system_prompt=coding_prompt,
    deps_type=CodeExecutionRequest,
    result_type=CodeExecutionResult
)

repl = PythonREPL()
@coding_agent.tool
def python_code_execution(
    ctx: RunContext,
    code: str,
):
    """Tool to perform code execution for python

    Args:
        ctx: The context.
        code: The python code to execute to perform tasks or answer questions

    Returns:
        str: The search results as a formatted string.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n```python\n{code}\n```\nWith Stdout: {result}"
    )
    return result_str

@researcher_agent.tool
async def search_web(ctx: RunContext, web_query: str) -> list[ResearchResult]:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """

    print ('\n * earch_web query:', web_query)

    search = DuckDuckGoSearchResults(output_format="list", max_results=5)
    web_results = search.invoke(web_query)

    results = []
    for item in web_results[:5]:
        title = item.get("title", "")
        description = item.get("snippet", "")
        url = item.get("link", "")
        results.append(ResearchResult(title=title, description=description, url=url))

    return results

@supervisor_agent.tool
async def researcher_tool(ctx: RunContext[None], query: ResearchQuery) -> ResearchOutput:
    result = await researcher_agent.run(query.query, deps=query)
    return result


@supervisor_agent.tool
async def coding_tool(ctx: RunContext[None], request: CodeExecutionRequest) -> CodeExecutionResult:
    result = await researcher_agent.run(request.description, deps=request)
    return result

'''


report_writer_agent = Agent(  
    'google-gla:gemini-2.0-flash',
    system_prompt=report_writer_prompt,
)





@supervisor_agent.tool
async def report_writer(ctx: RunContext[None], count: int) -> list[str]:
    r = await report_writer_agent.run(  
        f'Please generate {count} jokes.',
        usage=ctx.usage,  
    )
    return r.data  

result = supervisor_agent.run_sync(
    'Tell me a joke.',
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=300),
)

'''

result = supervisor_agent.run_sync(
    #'How many penguins live in Antarctica?',
    'How long does it take a cheetah to cross the golden gate bridge?',
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=20000),
)

for msg in result.all_messages():
    for part in msg.parts:
        if isinstance(part, ToolReturnPart):
            for m in part.content.all_messages():
                for p in m.parts:
                    print ()
                    print ('   ', p)
        else:
            print ()
            print (part)
