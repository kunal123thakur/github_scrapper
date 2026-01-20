"""
Routing functions for the workflow
"""
from typing import Literal
from models.state import AgentState


def route_from_chat(state: AgentState) -> Literal["planner", "end"]:
    """Route from chat: either to planner or end"""
    if state.get("reroute_to_planner"):
        print("🔀 Chat → Planner")
        return "planner"
    print("🔀 Chat → End")
    return "end"


def route_from_planner(state: AgentState) -> Literal["leetcode", "company", "hybrid"]:
    """Route from planner to appropriate agents"""
    intent_type = state["intent_classification"].get("intent_type", "leetcode")
    print(f"🔀 Planner → {intent_type}")
    return intent_type


def route_from_scheduler(state: AgentState) -> Literal["company", "end"]:
    """Route from scheduler: hybrid mode or end"""
    intent_type = state["intent_classification"].get("intent_type")
    has_company_questions = len(state.get("company_questions", [])) > 0
    
    if intent_type == "hybrid" and not has_company_questions:
        print("🔀 Scheduler → Company (hybrid mode)")
        return "company"
    
    print("🔀 Scheduler → End")
    return "end"
