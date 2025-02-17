from typing import Literal

from langchain import hub
from typing import Annotated

from langchain_openai import ChatOpenAI
from gradio import ChatMessage
import gradio as gr
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from pydantic import BaseModel, Field




# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

search_tool = DuckDuckGoSearchResults()

repl = PythonREPL()


# Define the input schema using Pydantic
class JokeInput(BaseModel):
    topic: str = Field(..., description="The topic of the joke (e.g., physics, AI, space).")

# Define the function that generates the joke
def generate_joke(topic: str) -> str:
    prompt = f"Tell me a funny joke about {topic}."
    return llm.predict(prompt)

# Create the LangChain tool
joke_tool = Tool(
    name="JokeGenerator",
    func=generate_joke,
    description="Generates a joke based on a given topic.",
    args_schema=JokeInput,
)

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


tools = [search_tool, python_repl_tool, joke_tool]
tools = [joke_tool]

trimmer = trim_messages(
    max_tokens=8192,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


graph = create_react_agent(llm, tools=tools)

sys_prompt = ('''You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a couple examples using notional tools:

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
Code:
```py
result = 5 + 3 + 1294.678
final_answer(result)
```<end_code>

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
```<end_code>
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".
              
Always execute the code with the python_repl_tool tool.

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.'''
)



print(sys_prompt)
lg_messages = [SystemMessage('Speak like a pirate, and use your joke when possible.')]

async def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    lg_messages.append(HumanMessage(prompt))

    async for event in graph.astream({"messages": lg_messages}):
        for e in event.values():
            msg = e["messages"][-1]

            lg_messages.append(msg)
            if len(msg.content) < 2:
                continue
            elif isinstance(msg, ToolMessage):
                messages.append(ChatMessage(role="assistant", content=msg.content, metadata={"title": f"🛠️  From tool {msg.name}:"}))
            else:
                messages.append(ChatMessage(role="assistant", content=msg.content))
            yield messages




with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
     # Function to clear chat history
    def clear_chat():
        return []

    with gr.Row():
        input_box = gr.Textbox(lines=1, label="Chat Message",submit_btn=True, scale=10)
        clear_btn = gr.Button("Clear Chat")  # Button to reset history

    # Clear chat when clicking "Clear Chat"
    clear_btn.click(clear_chat, [], chatbot)
    input_box.submit(interact_with_langchain_agent, [input_box, chatbot], [chatbot])

demo.launch()
