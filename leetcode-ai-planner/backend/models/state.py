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
    route_to_specialist: bool
    
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
    
    # DSA Specialist outputs
    dsa_concepts_covered: list
    dsa_complexity: dict
    dsa_follow_up: str
    dsa_confidence: str
    
    # 🆕 NEW: Context tracking (minimal addition)
    last_questions_shown: list  # Questions from last query
    context_topics: list  # Topics extracted by planner
    context_company: str  # Company from last query
    
    errors: Annotated[Sequence[str], operator.add]
    final_response: dict
