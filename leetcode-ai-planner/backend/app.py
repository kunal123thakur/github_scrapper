"""
FastAPI application entry point
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph.workflow import app_graph
from database.redis_store import memory_store
from database.vector_store import vector_store


app = FastAPI(title="Bhindi AI v3.0 - Modular Architecture")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
def chat(req: ChatRequest):
    """Main chat endpoint"""
    initial_state = {
        "user_message": req.message,
        "session_id": req.session_id,
        "errors": [],
        "reroute_to_planner": False
    }
    
    try:
        result = app_graph.invoke(initial_state)
        
        if not result.get("reroute_to_planner"):
            return {
                "success": True,
                "type": "chat",
                "response": result.get("chat_response"),
                "session_id": req.session_id
            }
        
        return {
            "success": True,
            "type": "execution",
            "intent": result["intent_classification"].get("intent_type"),
            "tools_selected": result.get("tools_selected", []),
            "schedule": result.get("combined_schedule"),
            "total_questions": (
                len(result.get("leetcode_questions", [])) + 
                len(result.get("company_questions", []))
            ),
            "session_id": req.session_id
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "leetcode_questions": len(vector_store.leetcode_df),
        "company_questions": len(vector_store.company_df),
        "version": "3.0-modular"
    }


@app.delete("/clear/{session_id}")
def clear_memory(session_id: str):
    """Clear chat history"""
    memory_store.clear(session_id)
    return {"success": True, "message": f"Cleared history for {session_id}"}


@app.get("/redis/test")
def test_redis():
    """Test Redis connection"""
    if not memory_store.redis:
        return {"connected": False}
    try:
        memory_store.redis.ping()
        return {"connected": True, "message": "Redis is working!"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
