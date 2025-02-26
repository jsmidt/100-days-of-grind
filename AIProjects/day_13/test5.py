############
#
#  MATH
#
############


from langgraph.func import task, entrypoint

# Define math tasks
@task
def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

@task
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

# Define Math Agent entrypoint
@entrypoint()
def math_agent(inputs: dict) -> float:
    """
    Math subgraph entrypoint.
    Expects inputs with 'operation' and numbers, e.g., {"operation": "add", "a": 2, "b": 3}.
    """
    op = inputs.get("operation")
    if op == "add":
        result = add(inputs["a"], inputs["b"]).result()       # perform addition
    elif op == "multiply":
        result = multiply(inputs["a"], inputs["b"]).result()  # perform multiplication
    else:
        raise ValueError(f"Unsupported operation: {op}")
    return result

############
#
#  WEATHER
#
############


# Define weather task
@task
def get_current_temperature(location: str) -> float:
    """Dummy function to get current temperature for a location."""
    dummy_temps = {"New York": 20.0, "London": 15.0, "Tokyo": 18.5}
    return dummy_temps.get(location, 25.0)  # default 25.0 if location not in dummy data

# Define Weather Agent entrypoint
@entrypoint()
def weather_agent(inputs: dict) -> float:
    """
    Weather subgraph entrypoint.
    Expects inputs with 'location', e.g., {"location": "London"}.
    Returns the current temperature in that location.
    """
    location = inputs.get("location")
    temperature = get_current_temperature(location).result()
    # In a more complex scenario, you might fetch conditions, format output, etc.
    return temperature



############
#
#  SUPERVISOR
#
############


@entrypoint()
def supervisor_agent(inputs: dict) -> str:
    """
    Supervisor agent entrypoint.
    Orchestrates calls to subgraphs based on the inputs or query.
    """
    query = inputs.get("query", "")      # e.g., "Double the temperature in London"
    location = inputs.get("location")    # e.g., "London"
    result_parts = {}

    # Determine needed sub-tasks (in a real scenario, parse the query to decide this)
    tasks_needed = []
    if location: 
        tasks_needed.append("weather")   # If a location is provided, likely need weather info
    if "double" in query.lower() or inputs.get("need_math"):
        tasks_needed.append("math")      # If query involves "double" (or explicitly needs math)

    # Iteratively call subgraph tools until all tasks are complete
    for task_name in tasks_needed:
        if task_name == "weather":
            # Call the Weather subgraph as a tool
            result_parts["weather"] = weather_agent.invoke({"location": location})
        elif task_name == "math":
            # Use the result of the weather subgraph in the math subgraph
            number = result_parts.get("weather")  # temperature fetched
            result_parts["math"] = math_agent.invoke({"operation": "multiply", "a": number, "b": 2})
        # (If more sub-tasks or different agents were needed, handle them similarly)

    # Combine/refine results into final answer
    if "weather" in result_parts and "math" in result_parts:
        final_answer = (f"The current temperature in {location} is {result_parts['weather']}°C, "
                        f"so double that is {result_parts['math']}°C.")
    elif "weather" in result_parts:
        final_answer = f"The current temperature in {location} is {result_parts['weather']}°C."
    elif "math" in result_parts:
        final_answer = f"The math result is {result_parts['math']}."
    else:
        final_answer = "No result."

    return final_answer


# Example usage (assuming the above definitions have been executed in a LangGraph context):

# Independent execution of subgraphs:
math_result = math_agent.invoke({"operation": "add", "a": 5, "b": 7})
print(f"Independent Math Agent result (5+7): {math_result}")
# Expected output: Independent Math Agent result (5+7): 12

weather_result = weather_agent.invoke({"location": "London"})
print(f"Independent Weather Agent result (London temp): {weather_result}°C")
# Expected output: Independent Weather Agent result (London temp): 15.0°C

# Combined execution via Supervisor:
final = supervisor_agent.invoke({
    "query": "Double the current temperature in London",
    "location": "London"
})
print(f"Supervisor Agent final output: {final}")
# Expected output: Supervisor Agent final output: The current temperature in London is 15.0°C, so double that is 30.0°C.