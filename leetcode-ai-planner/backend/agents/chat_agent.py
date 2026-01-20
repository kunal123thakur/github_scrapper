"""
Chat agent - Entry point with conversation memory
"""
import re
import json
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME
from database.redis_store import memory_store
from models.state import AgentState

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def chat_agent(state: AgentState) -> AgentState:
    """Smart chat agent with context memory"""
    session_id = state["session_id"]
    user_message = state["user_message"]
    
    history = memory_store.get(session_id)
    
    if history:
        conversation_context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in history[-10:]
        ])
        history_summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREVIOUS CONVERSATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conversation_context}

The user's NEW message is below. Use the conversation history above to maintain context.
"""
    else:
        history_summary = "This is the first message in this conversation."
    
    prompt = f"""
You are Bhindi AI, a smart DSA interview preparation assistant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CAPABILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Find LeetCode questions by topic, difficulty, tags
- Find company-specific interview questions (Google, Uber, Adobe, etc.)
- Create personalized study schedules
- Answer questions about DSA preparation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT AWARENESS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{history_summary}

IMPORTANT RULES FOR CONTEXT:
1. **Remember what the user told you** in previous messages
2. If user mentioned difficulty before, DON'T ask again
3. If user mentioned topic before, DON'T ask again  
4. If user mentioned company before, DON'T ask again
5. If user mentioned count before, DON'T ask again
6. Build upon previous conversation naturally
7. When you have enough info, reroute to planner

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REROUTING DECISION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set `reroute_to_planner = true` when you have ENOUGH information:
- Required: Either a topic OR a company OR "all questions"
- Optional but helpful: count, difficulty

Set `reroute_to_planner = false` when:
- Still gathering information
- User is asking general questions
- Request is unclear

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "response": "Your contextual reply to the user",
  "reroute_to_planner": true/false
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S NEW MESSAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_message}

Output ONLY valid JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            output = json.loads(json_match.group())
        else:
            output = json.loads(text)
        
        chat_response = output.get("response", "I'm here to help with DSA prep!")
        reroute = output.get("reroute_to_planner", False)
        
    except Exception as e:
        print(f"⚠️ Chat agent error: {e}")
        chat_response = "I'm here to help! Could you clarify what questions you're looking for?"
        reroute = False
    
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": chat_response})
    memory_store.set(session_id, history)
    
    state["chat_response"] = chat_response
    state["chat_output"] = chat_response
    state["reroute_to_planner"] = reroute
    state["chat_history"] = history
    
    print(f"💬 Chat: {len(history)} messages | Reroute: {reroute}")
    
    return state
