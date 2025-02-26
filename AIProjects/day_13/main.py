from rich.console import Console
from rich.markdown import Markdown

from better_prompt import call_llm

def main():
    console = Console()
    print("Hello from hello-uv!")
    md_code = """
# My Python Code

Here is some Python code:

```python
def hello_world():
     print("Hello, world!")
```
"""
    markdown_text = """
    # This is a title

    * List item 1
    * List item 2

    ```python
    def hello():
    print("Hello, world!")
    ```
    """
    md = Markdown(md_code)
    console.print(md)

    messages = call_llm('What is 3 times 4?')

    messages = call_llm('What is the weather in SF?')

    messages = call_llm('Write a report on monkeys?')

    print (messages)


if __name__ == "__main__":
    main()
