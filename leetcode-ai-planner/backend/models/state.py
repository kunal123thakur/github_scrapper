"""
State definitions for the agent graph
"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
import operator


class AgentState(TypedDict):
    user_message: str
    session_id: str
    chat_history: List[Dict[str, str]]
    
    # Chat agent outputs
    chat_response: Optional[str]
    chat_output: Optional[str]
    reroute_to_planner: bool
    
    # Planner outputs
    intent_classification: dict
    plan: List[str]
    tools_selected: List[str]
    
    # LeetCode results
    leetcode_interpretation: dict
    leetcode_questions: list
    
    # Company results
    company_interpretation: dict
    company_questions: list
    
    # Combined results
    combined_schedule: dict
    
    errors: Annotated[Sequence[str], operator.add]
    final_response: dict
