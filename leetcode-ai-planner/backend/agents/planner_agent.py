"""
Planner agent - Intent classification and parameter extraction
"""
import re
import json
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME
from models.state import AgentState

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def planner_agent(state: AgentState) -> AgentState:
    """Planner that extracts intent from conversation history"""
    user_message = state["user_message"]
    chat_output = state.get("chat_output", "")
    history = state.get("chat_history", [])
    
    all_user_messages = [
        msg['content'] for msg in history 
        if msg['role'] == 'user'
    ]
    
    full_user_context = " | ".join(all_user_messages[-5:])
    
    prompt = f"""
You are the PLANNER of a DSA preparation system.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL USER CONVERSATION CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User said (in order):
{full_user_context}

Latest message: {user_message}
Chat agent summary: {chat_output}

CRITICAL: Extract parameters from the ENTIRE conversation, not just the last message.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- leetcode_tool → General DSA questions
- company_tool → Company-specific questions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Look through ALL user messages to find:
- Topics: arrays, trees, graphs, DP, etc.
- Difficulty: easy, medium, hard
- Count: any number mentioned
- Company: google, uber, adobe, or "all companies"

Intent Classification:
- If user mentions company → "company"
- If user mentions general topics → "leetcode"  
- If both → "hybrid"

Examples:
- "easy" | "arrays" | "100 questions" → leetcode, topics=["arrays"], difficulty="easy", num=100
- "google" | "hard" | "10" → company, company_name="google", difficulty="hard", num=10
- "amazon and google" | "17 hard" → company, company_name="google, amazon", difficulty="hard", num=17

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "intent_type": "leetcode|company|hybrid",
  "confidence": 0.95,
  "reasoning": "brief explanation",
  "company_name": "google|uber|all|...",
  "num_questions": 100,
  "difficulty": "easy|medium|hard|any",
  "duration_days": 21,
  "topics": ["arrays"],
  "tools_selected": ["leetcode_tool"] or ["company_tool"] or both
}}

Output ONLY JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        intent = json.loads(json_match.group() if json_match else text)
        
        intent.setdefault("intent_type", "leetcode")
        intent.setdefault("num_questions", 15)
        intent.setdefault("duration_days", 21)
        intent.setdefault("difficulty", "any")
        intent.setdefault("topics", [])
        intent.setdefault("tools_selected", ["leetcode_tool"])
        
        intent["num_questions"] = int(intent["num_questions"])
        intent["duration_days"] = int(intent["duration_days"])
        
    except Exception as e:
        print(f"⚠️ Planner error: {e}")
        intent = {
            "intent_type": "leetcode",
            "num_questions": 15,
            "duration_days": 21,
            "difficulty": "any",
            "topics": [],
            "tools_selected": ["leetcode_tool"]
        }
    
    state["intent_classification"] = intent
    state["tools_selected"] = intent["tools_selected"]
    
    print(f"🎯 Planner: {intent['intent_type']} | {intent['num_questions']} questions")
    
    return state
