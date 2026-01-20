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
    route_to_specialist: bool  # NEW
    
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
    
    # DSA Specialist outputs (NEW)
    dsa_concepts_covered: list
    dsa_complexity: dict
    dsa_follow_up: str
    dsa_confidence: str
    
    errors: Annotated[Sequence[str], operator.add]
    final_response: dict
