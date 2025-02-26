# Schema for structured output
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import Literal, Optional
from langchain_core.tools import tool

from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode





llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


class BetterQuery(BaseModel):
    better_query: str = Field(..., description="Query optimized for the next agent's use.")
    next_agent: Optional[Literal["search_web", "write_report"]] = Field(
        None, description="Next agent to route to: 'search_web' for internet searches, 'write_report' for writing report queries, or None if not applicable."
    )


system_message = """
You are a query-routing and optimization agent. Your task is to analyze the user's prompt and return a structured JSON object that adheres strictly to the following schema:

{
  "better_query": "<optimized query for the next agent>",
  "next_agent": "<'search_web' | 'write_report' | null>"
}

Instructions:
1. Examine the user's prompt.
2. If the prompt involves doing a web search (e.g., asking for up-to-date information, current news, or any query best answered by retrieving web data), set "next_agent" to "search_web".
3. If the prompt involves writing a report (e.g., a request for a detailed summary, analysis, or comprehensive report on a topic), set "next_agent" to "write_report".
4. For any other prompt, set "next_agent" to null.
5. Optimize or rephrase the user's prompt to be more effective for the chosen next agent, and assign it to "better_query".
6. Return only the JSON object with no additional text or commentary.

Examples:
- For the user query "What is the latest NASA launch schedule?", your output should be:
{
  "better_query": "Get the latest NASA launch schedule",
  "next_agent": "search_web"
}
- For the user query "Write a detailed report on renewable energy trends", your output should be:
{
  "better_query": "Compose a detailed report on renewable energy trends",
  "next_agent": "write_report"
}
- For any other prompt (e.g., "Tell me a joke"), output:
{
  "better_query": "Tell me a joke",
  "next_agent": null
}

Ensure your final answer is valid JSON that exactly matches the schema.
"""


@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny!"
    elif "boston" in location.lower():
        return "It's rainy!"
    else:
        return f"I am not sure what the weather is in {location}"

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

tools = [get_weather, multiply]

tools_by_name = {tool.name: tool for tool in tools}

@task
def call_model(messages):
    """Call model with a sequence of messages."""
    response = llm.bind_tools(tools).invoke(messages)
    return response

@task
def call_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])

@entrypoint()
def router_agent(messages):
    llm_response = call_model(messages).result()
    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]

        # Append to message list
        messages = add_messages(messages, [llm_response, *tool_results])

        # Call model again
        llm_response = call_model(messages).result()

    return llm_response

def call_llm(prompt):


    '''

    messages = [
        SystemMessage(system_message),
        HumanMessage(prompt)
    ]

    structured_llm = llm.with_structured_output(BetterQuery)

    output = structured_llm.invoke(messages)

    print (output)
    '''

    tool_node = ToolNode(router_agent)

    print (tool_node)

    '''
    for step in router_agent.stream(messages):
        for task_name, message in step.items():
            #print (task_name)
            if task_name == "agent":
                continue  # Just print task updates
            #print(f"\n{task_name}:")
            #print (message.content)
            message.pretty_print()

    
    user_message = {"role": "user", "content": "What's the weather in san francisco?"}
    user_message = {"role": "user", "content": "What is 3 times 4?"}

    print(user_message)

    for step in agent.stream([user_message]):
        for task_name, message in step.items():
            #print (task_name)
            if task_name == "agent":
                continue  # Just print task updates
            print(f"\n{task_name}:")
            print (message.content)
            #message.pretty_print()
    
    # Augment the LLM with schema for structured output
    structured_llm = llm.with_structured_output(BetterQuery)

    messages = [
        SystemMessage(system_message),
        HumanMessage(prompt)
    ]

    # Invoke the augmented LLM
    output = structured_llm.invoke(messages)

    print (output)



    # Augment the LLM with tools
    if output.step == "multiply_tool":
        llm_with_tools = llm.bind_tools([multiply])
        # Invoke the LLM with input that triggers the tool call
        msg = llm_with_tools.invoke(output.search_query)

        print ('AI Message')
        print (msg)
        messages.append(msg)

        # Get the tool call
        for tool_call in msg.tool_calls:
            print ('Tool Message')
            tool_msg = multiply.invoke(tool_call)
            messages.append(tool_msg)

            print (tool_msg)


    return messages
    
    # Define a tool
    def multiply(a: int, b: int) -> int:
        return a * b

    # Augment the LLM with tools
    llm_with_tools = llm.bind_tools([multiply])

    # Invoke the LLM with input that triggers the tool call
    msg = llm_with_tools.invoke("What is 2 times 3?")

    # Get the tool call
    msg.tool_calls
    '''
