from pydantic import BaseModel
import threading
import queue
import sys

# Dummy decorators for tasks and entrypoints (for demonstration purposes)
def task(func=None, **kwargs):
    """Decorator to mark a function as a task."""
    def decorator(f):
        return f
    return decorator(func) if func else decorator

def entrypoint(func=None, **kwargs):
    """Decorator to mark an agent's entrypoint."""
    def decorator(f):
        return f
    return decorator(func) if func else decorator

# Pydantic model for structured running notes
class RunningNotes(BaseModel):
    location: str = None
    temperature: float = None
    factor_a: float = None
    factor_b: float = None
    product: float = None
    steps: list[str] = []  # log of steps/actions taken

    def add_step(self, info: str):
        """Add a descriptive step to the log."""
        self.steps.append(info)

@task
def fetch_weather(location: str) -> float:
    """
    Task: Fetch weather information (temperature) for the given location.
    (In a real implementation, this might call an API. Here we return a dummy value.)
    """
    # For demonstration, return a fixed temperature (°F)
    dummy_temperature = 68.0  # e.g., 68°F as the current temperature
    return dummy_temperature

@entrypoint()
def weather_agent(location: str, notes: RunningNotes) -> RunningNotes:
    """
    Weather Agent: Retrieves the weather for a location and updates the notes.
    """
    temperature = fetch_weather(location)
    # Update structured notes
    notes.location = location
    notes.temperature = temperature
    notes.add_step(f"Weather agent: The temperature in {location} is {temperature}°F.")
    return notes

@task
def multiply_numbers(a: float, b: float) -> float:
    """
    Task: Multiply two numbers and return the result.
    """
    return a * b

@entrypoint()
def math_agent(a: float, b: float, notes: RunningNotes) -> RunningNotes:
    """
    Math Agent: Performs a multiplication and updates the notes.
    """
    result = multiply_numbers(a, b)
    # Update structured notes
    notes.factor_a = a
    notes.factor_b = b
    notes.product = result
    notes.add_step(f"Math agent: Calculated {a} * {b} = {result}.")
    return notes

@entrypoint()
def report_writer(notes: RunningNotes) -> str:
    """
    Report Writing Agent: Summarizes the notes into a Markdown report.
    (Here we simulate using an LLM like ChatGoogleGenerativeAI to format the report.)
    """
    # Create a markdown-formatted report from the notes
    report_lines = []
    report_lines.append(f"- **Location:** {notes.location}")
    report_lines.append(f"- **Temperature:** {notes.temperature}°F")
    report_lines.append(f"- **Multiplication:** {notes.factor_a} × {notes.factor_b} = {notes.product}")
    # Combine lines into a Markdown string with a heading
    markdown_report = "# Final Report\n\n" + "\n".join(report_lines) + "\n"
    return markdown_report

@entrypoint()
def supervisor_agent():
    """
    Supervisor Agent: Coordinates the workflow, including human-in-the-loop review.
    """
    # Initialize running notes
    notes = RunningNotes()
    # 1. Call Weather Agent subgraph
    notes = weather_agent("Española, New Mexico, US", notes)
    # 2. Call Math Agent subgraph (for example, multiply 7 by 6)
    notes = math_agent(7, 6, notes)
    # 3. Human-in-the-loop review step
    print("Structured Notes (for review):")
    print(notes.model_dump_json(indent=2))  # display notes in JSON format for clarity

    # Prompt the user for approval with a timeout of 60 seconds
    print("\nPlease review the notes above.")
    print("Approve the notes to generate the report? (y/n) [Default is 'y' after 1 minute]: ", end="")
    sys.stdout.flush()

    # Use a thread to handle input with timeout
    user_input_queue = queue.Queue()
    def ask_user():
        answer = sys.stdin.readline().strip()  # read input (this will block until a newline)
        user_input_queue.put(answer)

    thread = threading.Thread(target=ask_user, daemon=True)
    thread.start()
    thread.join(timeout=60.0)  # wait for up to 60 seconds for user input

    approved = True  # default to proceed
    if thread.is_alive():
        # No input received in 1 minute
        print("\n[No response within 60 seconds. Proceeding with report generation.]")
    else:
        # Input received; get it from the queue
        user_answer = user_input_queue.get().lower()
        if user_answer in ["n", "no", "reject"]:
            approved = False
        # Treat any other input (including 'y' or empty) as approval
        if approved:
            print("\n[User approved the notes. Generating report...]")
        else:
            print("\n[User rejected the notes. Halting execution.]")

    # 4. If approved, call the Report Writing agent; otherwise, exit without generating a report
    if not approved:
        return  # halt execution
    report = report_writer(notes)
    # 5. Output the final Markdown report
    print("\nGenerated Report:\n")
    print(report)

# Entry point: run the supervisor agent (if this script is executed directly)
if __name__ == "__main__":
    supervisor_agent()