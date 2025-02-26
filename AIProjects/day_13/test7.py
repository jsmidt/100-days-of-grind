from pydantic import BaseModel
from langgraph.func import entrypoint, task
from langchain_google_genai import ChatGoogleGenerativeAI

# Pydantic model to hold running notes (shared state)
class RunningNotes(BaseModel):
    location: str = None
    temperature: float = None
    multiplier: float = None
    product: float = None

# Define tasks for core functionalities
@task
def fetch_temperature(location: str) -> float:
    """Simulate fetching the current temperature for the given location."""
    # In a real system, call a weather API here.
    print(f"WeatherAgent: Getting weather for {location}...")
    temperature = 72.0  # placeholder for demo
    print(f"WeatherAgent: The temperature in {location} is {temperature}°F.")
    return temperature

@task
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    product = a * b
    print(f"MathAgent: The product of {a} and {b} is {product}.")
    return product

# Define entrypoint agents
@entrypoint()
def weather_agent(location: str) -> float:
    """Weather agent entrypoint: uses fetch_temperature task to get weather data."""
    return fetch_temperature(location)  # returns temperature

@entrypoint()
def math_agent(x: float, y: float) -> float:
    """Math agent entrypoint: uses multiply task to compute a product."""
    return multiply(x, y)  # returns product

@entrypoint()
def joke_agent() -> str:
    """Joke agent entrypoint: generates a joke using Google's Gemini LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    prompt = "Tell me a short, funny joke."
    joke = llm.predict(prompt)  # get a joke from the LLM
    return joke

@entrypoint()
def report_writer(notes: RunningNotes) -> str:
    """Report writer agent: produces a markdown report from the notes."""
    # Compose a simple Markdown report using data from the notes
    report_md = "# Weather and Math Report\n"
    report_md += f"- **Location**: {notes.location}\n"
    report_md += f"- **Temperature**: {notes.temperature} °F\n"
    report_md += f"- **Multiplier**: {notes.multiplier}\n"
    report_md += f"- **Product**: {notes.product}\n"
    return report_md

@entrypoint()
def supervisor() -> None:
    """Supervisor agent: orchestrates weather, math, and decides between joke or report."""
    notes = RunningNotes()  # initialize structured notes

    # 1. Call Weather Agent subgraph to get temperature
    notes.location = "Española, New Mexico"  # example location (could be dynamic)
    notes.temperature = weather_agent(notes.location)  # execute weather_agent
    # 2. Call Math Agent subgraph to perform a multiplication (e.g., double the temperature)
    notes.multiplier = 2.0
    notes.product = math_agent(notes.temperature, notes.multiplier)

    # 3. Human-in-the-loop: prompt user for choice with timeout
    import sys, select
    prompt = "Would you like a joke or a report? (j/r) [Default is 'r' after 60 seconds]: "
    print(prompt, end="", flush=True)
    user_choice = None
    try:
        # Wait for input on stdin for up to 60 seconds
        rlist, _, _ = select.select([sys.stdin], [], [], 60)
        if rlist:
            # If user provided input within 60 seconds, read it
            user_input = sys.stdin.readline().strip().lower()
            if user_input in ("j", "r"):
                user_choice = user_input
    except Exception as e:
        # In case select is not supported or any error, default to report
        user_choice = None

    # Default to 'r' (report) if no valid choice was made
    if user_choice is None:
        user_choice = "r"

    # 4. Invoke the chosen agent
    if user_choice == "j":
        # User chose joke: call joke_agent and print the joke
        joke = joke_agent()
        print("\n[Joke] " + joke)
    else:
        # User chose report (or timed out/invalid input): call report_writer and print report
        report = report_writer(notes)
        print("\n" + report)

# Run the supervisor workflow (if this script is executed directly)
if __name__ == "__main__":
    supervisor()