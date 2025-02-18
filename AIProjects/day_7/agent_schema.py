# agent_schemas.py
"""
This module defines the Pydantic data classes for structured inputs and outputs 
of the multi-agent system.

Agents:
1. Researcher Agent - Requires search queries, returns summarized results.
2. Coding Agent - Requires code tasks, returns structured computation results.
3. Report Writer Agent - Requires structured data, returns formatted reports.

Each agent strictly follows Pydantic schemas for consistent and structured communication.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any


# ---------------------------
# Researcher Agent Schemas
# ---------------------------
class ResearchQuery(BaseModel):
    """Input schema for the Researcher Agent."""
    query: str = Field(..., description="A subject or question to search for.")
    additional_keywords: Optional[List[str]] = Field(None, description="Additional keywords to refine the search.")
    max_results: int = Field(5, description="Maximum number of results to retrieve.")
    require_recent: bool = Field(False, description="Whether to prioritize recent sources.")


class ResearchResult(BaseModel):
    """Structured output from the Researcher Agent."""
    title: str = Field(..., description="Title of the research source.")
    description: str = Field(..., description="Brief summary of the content.")
    url: HttpUrl = Field(..., description="Link to the full source.")


class ResearchOutput(BaseModel):
    """Aggregated research findings."""
    query: str = Field(..., description="The query searched.")
    results: List[ResearchResult]
    notes: Optional[str] = Field(None, description="Additional observations or warnings about the findings.")


# ---------------------------
# Coding Agent Schemas
# ---------------------------
class CodeExecutionRequest(BaseModel):
    """Input schema for the Coding Agent."""
    description: str = Field(..., description="Description of the computation or code task.")
    #code: str = Field(..., description="Python code snippet to execute.")
    expected_output_type: Optional[str] = Field(None, description="Expected format of the output (e.g., 'number', 'table').")


class CodeExecutionResult(BaseModel):
    """Structured output from the Coding Agent."""
    success: bool = Field(..., description="Indicates if the execution was successful.")
    output: Any = Field(..., description="The result of the computation.")
    error_message: Optional[str] = Field(None, description="Error message if execution failed.")


# ---------------------------
# Report Writer Agent Schemas
# ---------------------------
class ReportRequest(BaseModel):
    """Input schema for the Report Writer Agent."""
    title: str = Field(..., description="Title of the report.")
    summary: str = Field(..., description="Overview of the findings.")
    research_findings: List[ResearchResult] = Field(..., description="Research results to include.")
    computation_results: Optional[Dict[str, CodeExecutionResult]] = Field(None, description="Processed results from the Coding Agent.")


class ReportOutput(BaseModel):
    """Structured output from the Report Writer Agent."""
    formatted_report: str = Field(..., description="The final structured report as a text document.")
    sections: Dict[str, str] = Field(..., description="A dictionary of report sections for easy reference.")


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Example Research Query
    research_input = ResearchQuery(
        topic="Quantum Computing Applications",
        additional_keywords=["AI", "cryptography"],
        max_results=3,
        require_recent=True
    )
    
    print(research_input.json(indent=4))  # Pretty print structured input