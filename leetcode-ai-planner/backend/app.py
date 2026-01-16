from fastapi import FastAPI, HTTPException
import sys
sys.path.append("..")

from backend.schemas import ChatRequest
from backend.dataset_loader import load_dataset
from backend.validator import validate_intent

from backend.agents.intent_agent import extract_intent
from backend.agents.planner_agent import plan_distribution
from backend.agents.filter_agent import filter_questions
from backend.agents.scheduler_agent import schedule
from backend.agents.response_agent import build_response

app = FastAPI()
df = load_dataset()

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        intent = extract_intent(req.message)
        validate_intent(intent)

        plan = plan_distribution(intent)
        filtered = filter_questions(df, plan)
        scheduled = schedule(filtered, plan["duration_days"])

        return build_response(scheduled)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
