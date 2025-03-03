from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessageChunk, SystemMessage, HumanMessage
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages



from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

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


tools = [get_weather]


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
def agent(messages):
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

tool_desc = "Get weather tool"
tool_names = "get_weather"
system_prompt = f"""
'You are designed to help with a variety of tasks, from answering questions to 
providing summaries to other types of analyses.\n\n## Tools\n\nYou have access
to a wide variety of tools. You are responsible for using the tools in any
sequence you deem appropriate to complete the task at hand.\nThis may require
breaking the task into subtasks and using different tools to complete each
subtask.\n\nYou have access to the following tools:\n{tool_desc}\n\n\n## Output
Format\n\nPlease answer in the same language as the question and use the 
following format:\n\n```\nThought: The current language of the user is:
(user\'s language). I need to use a tool to help me answer the
question.\nAction: tool name (one of {tool_names}) if using a tool.\nAction
Input: the input to the tool, in a JSON format representing the kwargs (e.g.
{{"input": "hello world", "num_beams": 5}})\n```\n\nPlease ALWAYS start with a
Thought.\n\nNEVER surround your response with markdown code markers. You may
use code markers within your response if you need to.\n\nPlease use a valid
JSON format for the Action Input. Do NOT do this {{\'input\': \'hello world\',
\'num_beams\': 5}}.\n\nIf this format is used, the tool will respond in the
following format:\n\n```\nObservation: tool response\n```\n\nYou should keep
repeating the above format till you have enough information to answer the
question without using any more tools. At that point, you MUST respond in one
of the following two formats:\n\n```\nThought: I can answer without using any
more tools. I\'ll use the user\'s language to answer\nAnswer: [your answer here
(In the same language as the user\'s question)]\n```\n\n```\nThought: I cannot
answer the question with the provided tools.\nAnswer: [your answer here (In the
same language as the user\'s question)]\n```\n\n## Current
Conversation\n\nBelow is the current conversation consisting of interleaving
human and assistant messages.\n'"""


user_message = [SystemMessage(system_prompt), HumanMessage("Who is Albert Einstein?")]
user_message = [SystemMessage(system_prompt), HumanMessage("What is the weather in SF?")]

for step in agent.stream(user_message,stream_mode=['updates','messages']):
    stream_type, stream_value = step
    #print (stream_type)
    #if stream_type == "messages":
    #    for msg in stream_value:
    #        if isinstance(msg, AIMessageChunk):
    #            print (msg.content,end='')
    if stream_type == "updates":
        for task_name, message in stream_value.items():
            if task_name == "agent":
                continue  # Just print task updates
            #print(f"\n{task_name}:")
            message.pretty_print()