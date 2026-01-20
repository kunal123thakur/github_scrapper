"""
LangGraph workflow definition
"""
from langgraph.graph import StateGraph, END
from models.state import AgentState
from agents.chat_agent import chat_agent
from agents.planner_agent import planner_agent
from agents.leetcode_agent import leetcode_search_agent
from agents.company_agent import company_search_agent
from agents.scheduler_agent import scheduler_agent
from graph.routes import route_from_chat, route_from_planner, route_from_scheduler


def build_workflow():
    """Builds and compiles the agent workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("chat", chat_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("leetcode", leetcode_search_agent)
    workflow.add_node("company", company_search_agent)
    workflow.add_node("scheduler", scheduler_agent)
    
    # Entry point
    workflow.set_entry_point("chat")
    
    # Chat routing
    workflow.add_conditional_edges(
        "chat",
        route_from_chat,
        {
            "planner": "planner",
            "end": END
        }
    )
    
    # Planner routing
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "leetcode": "leetcode",
            "company": "company",
            "hybrid": "leetcode"
        }
    )
    
    # LeetCode → scheduler
    workflow.add_edge("leetcode", "scheduler")
    
    # Company → scheduler
    workflow.add_edge("company", "scheduler")
    
    # Scheduler routing
    workflow.add_conditional_edges(
        "scheduler",
        route_from_scheduler,
        {
            "company": "company",
            "end": END
        }
    )
    
    return workflow.compile()


# Singleton instance
app_graph = build_workflow()
