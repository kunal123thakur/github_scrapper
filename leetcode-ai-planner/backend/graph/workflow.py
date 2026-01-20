"""
LangGraph workflow with DSA Specialist
"""
from langgraph.graph import StateGraph, END
from models.state import AgentState
from agents.chat_agent import chat_agent
from agents.planner_agent import planner_agent
from agents.leetcode_agent import leetcode_search_agent
from agents.company_agent import company_search_agent
from agents.scheduler_agent import scheduler_agent
from agents.dsa_specialist_agent import dsa_specialist_agent
from graph.routes import route_from_chat, route_from_planner, route_from_scheduler


def build_workflow():
    """Build workflow with DSA specialist"""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("chat", chat_agent)
    workflow.add_node("specialist", dsa_specialist_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("leetcode", leetcode_search_agent)
    workflow.add_node("company", company_search_agent)
    workflow.add_node("scheduler", scheduler_agent)
    
    # Entry point
    workflow.set_entry_point("chat")
    
    # Chat routing (3-way)
    workflow.add_conditional_edges(
        "chat",
        route_from_chat,
        {
            "specialist": "specialist",
            "planner": "planner",
            "end": END
        }
    )
    
    # Specialist ends conversation
    workflow.add_edge("specialist", END)
    
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
    
    # Execution flow
    workflow.add_edge("leetcode", "scheduler")
    workflow.add_edge("company", "scheduler")
    
    workflow.add_conditional_edges(
        "scheduler",
        route_from_scheduler,
        {
            "company": "company",
            "end": END
        }
    )
    
    return workflow.compile()


app_graph = build_workflow()
