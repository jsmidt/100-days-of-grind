from rich.console import Console
from rich.markdown import Markdown

console = Console()

md_code = """
# My Python Code

Here is some Python code:

```python
def hello_world():
    print("Hello, world!")
```
"""

markdown = Markdown(md_code)
console.print(markdown)
