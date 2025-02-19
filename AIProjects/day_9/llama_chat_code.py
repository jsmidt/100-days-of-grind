from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from tavily import AsyncTavilyClient
from gradio import ChatMessage
import gradio as gr
from llama_index.core.workflow import Context

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.utils.workflow import draw_all_possible_flows

llm = Gemini(model="models/gemini-2.0-flash")
verbose = False

sysprompt = '''
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given a tool that can execute Python code. You also have a ResearchAgent to assist you with internet searches.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python, and hand it off to your CodeInterpreterTool.
In the end you have to return a final answer.

ALWAYS EXECUTE THE PYTHON CODE YOU GENERATE CALLING YOUR TOOL AND USE THE RESULT IN YOUR ANSWER.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```

Hand off that code to the CodeInterpreterTool.

Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person.
Code:
```py
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
```

Hand off that code to the CodeInterpreterTool.

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
Code:
```py
result = 5 + 3 + 1294.678
```

Hand off that code to the CodeInterpreterTool to obtain the final answer.

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
```

Hand off that code to the CodeInterpreterTool to obtain the final answer.

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.  I will ask the ResearchAgent to do a web search.

Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
Code:

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the Research Agent to get the population of both cities.

Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
Final answer: Shanghai

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the Research Agent to find this information.  

Observation: after calling the search agent, I can wrote this code:

Hand off that code to the CodeInterpreterTool to obtain the final answer.

Code:
```py
pope_age_wiki = wiki(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
```

Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 88 ** 0.36
```

The final answer is pope_current_age

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.
11. Always execute the python code you generate, and use the output in your final answer.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
'''











##############################
#
# Supervisor Agent
#
##############################

supervisor_agent = ReActAgent(
    name="SupervisorAgent",
    description="Supervisor agent to oversee answering questions by orchestrating other agents.",
    system_prompt=(
        "You are the SupervisorAgent tasked with answering questions. You always respond as a pirate and in Markdown.\n"
        "You have agents at your disposal. Always use them whenever you determine it's helpful."
        "The first step you must take is always restate the task given in your own words writing a better prompt than provided"
        "Next, you must explain in words how each agent - the ResearchAgent and the CodingAgent - could assist in the task. "
        "Then you must prepare a step-by-step plan how through using each agent you can complete the task.\n\n"
        "If any type of information is useful, make sure to task the ResearchAgent to do an appropriate web search."
        "If there is any task the would benefit from running a Python program,task the CodingAgent to do so.\n\n"
        "As an example, if you are asked to calculate how long it would take for a cheetah to cross the golden gate bridge,"
        "you would devise a plan such as:"
        "Step 1: Find out how fast a cheetah runs using the ResearchAgent.\n" 
        "Step 2: Find out the distance of the golden gate bridge using the ResearchAgent.\n"
        "Step 3: Write and execute a python program using the CodingAgent to calculate the time required.\n"
        "Step 4: Give the user the final answer.\n\n"
        "Here is another example. Let's say the user asks to make a plot of the UK's GDP from 2020 - 2024."
        "Your plan might look like:\n"
        "Step 1: Find out the UK's GDP using the ResearchAgent.\n"
        "Step 2: Tell the coding agent to use matplotlib to plot this data and save it as myplot.png\n"
        "Step 3: Tell the user in Markdown what you found about the GDP, then inform the user where to find the created plot.\n\n"
        "You should always be the agent that gives the final answer."
    ),
    llm=llm,
    tools=[],
    can_handoff_to=["ResearchAgent", "CodingAgent"],
)



##############################
#
# Research Agent
#
##############################

code_spec = CodeInterpreterToolSpec()

coding_agent = FunctionAgent(
    name="CodingAgent",
    description="Useful agent for executing python code to answer questions.",
    system_prompt=sysprompt,
    #system_prompt=(
    #    "You are the CodingAgent that can write and exeute any python code to answer quetions. "
    #    "You have all of the system's python libraries at your disposal."
    #    "Your job is to inform the SupervisorAgent. Let the SupervisorAgent give the final answer."
    #),
    llm=llm,
    tools=code_spec.to_tool_list(),
    can_handoff_to=["ResearchAgent"],
)



##############################
#
# Coding Agent
#
##############################

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    return str(await client.search(query))


research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful agent for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic. "
        "Your job is only to find new information and inform the CodingAgent. Let the CodingAgent give the final answer."
        "You also have access to a CodingAgent who should answer questions best done by a python program. Such as doing math."
        
    ),
    llm=llm,
    tools=[search_web],
    can_handoff_to=["CodingAgent"],
)




##############################
#
#  Agent WorkflowResearch Agent
#
##############################

#agent_workflow = AgentWorkflow(agents=[supervisor_agent, research_agent, coding_agent], root_agent=supervisor_agent.name)
agent_workflow = AgentWorkflow(agents=[research_agent, coding_agent], root_agent=coding_agent.name)

ctx = Context(agent_workflow)


async def interact_with_llamaindex_agent(prompt, messages):
    # Need to add user prompt to gradio chat history
    messages.append(ChatMessage(role="user", content=prompt))

    # Return prompt message to update gradio App
    yield messages

    # Initialize assistant messages
    # async with agent.run_stream(prompt, message_history=msgs) as result:
    handler = agent_workflow.run(user_msg=prompt, ctx=ctx)
    current_agent = None
    not_streaming = True
    async for event in handler.stream_events():

        # Print what agent we are on
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            # Only create a new chat message if we aren't in one
            if not_streaming:
                messages.append(ChatMessage(role="assistant", content=""))
                not_streaming = False
            current_agent = event.current_agent_name
            partial_message = ''
            partial_message += f"\n{'='*50}"
            partial_message += f"🤖 Agent: {current_agent}"
            partial_message += f"{'='*50}\n\n"
            messages[-1].content += partial_message
            yield messages
        
        # Stream the Agent's tokens if it's in streaming moce
        elif isinstance(event, AgentStream):
            # Only create a new chat message if we aren't in one
            if not_streaming:
                messages.append(ChatMessage(role="assistant", content=""))
                not_streaming = False
            messages[-1].content += event.delta
            yield messages

        # Speical output if the Agent is thking to use a tool or handoff to other agent
        elif isinstance(event, AgentOutput):
            if event.tool_calls:
                not_streaming = True
                messages.append(ChatMessage(role="assistant", content="", metadata={"title": '🧠 Thinking'}))
                partial_message = ''
                partial_message += f"📖 Planning to use tools: {[call.tool_name for call in event.tool_calls]}\n"
                messages[-1].content += partial_message
                yield messages   
                
        # Print the results of the tool call in a special tool box
        elif isinstance(event, ToolCallResult):
            not_streaming = True
            messages.append(ChatMessage(role="assistant", content="", metadata={"title": f'New action: {event.tool_name}'}))
            partial_message = ''
            partial_message += f"🔧 Tool Result ({event.tool_name}):\n"
            partial_message += f"  Arguments: {event.tool_kwargs}\n"
            partial_message += f"  Output: {event.tool_output}\n"
            messages[-1].content += partial_message
            yield messages
        # If a new tool is called, let user know in a special box

        # Print to terminal if verbose. 
        if verbose:
            if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
            ):
                current_agent = event.current_agent_name
                print(f"\n{'='*50}")
                print(f"🤖 Agent: {current_agent}")
                print(f"{'='*50}\n")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                if event.tool_calls:
                    print(
                        "🛠️  Planning to use tools:",
                        [call.tool_name for call in event.tool_calls],
                    )
            elif isinstance(event, ToolCallResult):
                print(f"🔧 Tool Result ({event.tool_name}):")
                print(f"  Arguments: {event.tool_kwargs}")
                print(f"  Output: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"🔨 Calling Tool: {event.tool_name}")
                print(f"  With arguments: {event.tool_kwargs}")




with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LlamaIndex 🦙 and see its thoughts 💭 and tools 🛠")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input_box = gr.Textbox(lines=1, label="Chat Message")
    input_box.submit(interact_with_llamaindex_agent, [input_box, chatbot], [chatbot])

draw_all_possible_flows(agent_workflow, filename="multi_step_workflow.html")
demo.launch()
