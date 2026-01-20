"""
Scheduler agent - Creates daily schedule
"""
from datetime import date, timedelta
from models.state import AgentState


def scheduler_agent(state: AgentState) -> AgentState:
    """Creates schedule for questions"""
    intent = state["intent_classification"]
    intent_type = intent.get("intent_type")
    duration = int(intent.get("duration_days", 21))
    
    all_questions = []
    
    if intent_type in ["leetcode", "hybrid"]:
        all_questions.extend(state.get("leetcode_questions", []))
    
    if intent_type in ["company", "hybrid"]:
        all_questions.extend(state.get("company_questions", []))
    
    if not all_questions:
        state["combined_schedule"] = {}
        return state
    
    all_questions = sorted(all_questions, key=lambda x: -x.get("similarity_score", 0))
    
    start = date.today()
    schedule_map = {}
    
    for i, q in enumerate(all_questions):
        day = start + timedelta(days=i % duration)
        
        if q.get("source") == "leetcode":
            question_entry = {
                "title": q.get("task_id", "").replace("-", " ").title(),
                "difficulty": q.get("difficulty", "").title(),
                "tags": q.get("tags", [])[:3],
                "url": f"https://leetcode.com/problems/{q.get('task_id', '')}/",
                "source": "LeetCode",
                "similarity": round(q.get("similarity_score", 0), 3)
            }
        else:
            question_entry = {
                "title": q.get("Question_Name", ""),
                "difficulty": q.get("Difficulty", "").title(),
                "company": q.get("Company", "").title(),
                "time_to_solve": q.get("Time_to_Solve", ""),
                "url": q.get("Problem_Link", ""),
                "source": f"{q.get('Company', '').title()} Interview",
                "similarity": round(q.get("similarity_score", 0), 3)
            }
        
        schedule_map.setdefault(str(day), []).append(question_entry)
    
    state["combined_schedule"] = schedule_map
    print(f"📅 Scheduled {len(all_questions)} questions over {duration} days")
    
    return state
