"""
Chat agent with DSA specialist routing
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
    """Smart chat agent with routing"""
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
"""
    else:
        history_summary = "First message in this conversation."
    
    prompt = f"""
You are Bhindi AI's routing agent. Analyze the user's intent and route appropriately.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{history_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Route to DSA SPECIALIST** (`route_to_specialist = true`) when user is asking about:
- Conceptual questions: "explain", "what is", "how does", "why"
- Learning: "teach me", "help me understand", "difference between"
- Theory: algorithms, data structures, complexity, patterns
- Problem-solving strategies or approaches
- Examples: "explain DFS", "what is DP", "how does binary search work"

**Route to PLANNER** (`reroute_to_planner = true`) when user wants:
- Practice questions: "give me questions", "practice problems"
- Study schedule: "create schedule", "plan my prep"
- Specific count + topic: "10 array questions", "hard DP problems"
- Company-specific: "Google questions", "Amazon interview prep"

**Stay in CHAT** (both false) when:
- Greeting or casual chat
- Unclear intent - ask clarification
- Gathering more information

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S MESSAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_message}

Output JSON:
{{
  "response": "Your reply (only if staying in chat)",
  "reroute_to_planner": true/false,
  "route_to_specialist": true/false,
  "reasoning": "1-2 sentence explanation"
}}
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        output = json.loads(json_match.group() if json_match else text)
        
        chat_response = output.get("response", "How can I help you today?")
        reroute_planner = output.get("reroute_to_planner", False)
        route_specialist = output.get("route_to_specialist", False)
        reasoning = output.get("reasoning", "")
        
    except Exception as e:
        print(f"⚠️ Chat agent error: {e}")
        chat_response = "I'm here to help! Are you looking to practice questions or learn DSA concepts?"
        reroute_planner = False
        route_specialist = False
        reasoning = ""
    
    # Only save if staying in chat
    if not reroute_planner and not route_specialist:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": chat_response})
        memory_store.set(session_id, history)
    
    state["chat_response"] = chat_response
    state["chat_output"] = chat_response
    state["reroute_to_planner"] = reroute_planner
    state["route_to_specialist"] = route_specialist
    state["chat_history"] = history
    
    print(f"💬 Chat: Specialist={route_specialist}, Planner={reroute_planner}")
    print(f"   Reasoning: {reasoning}")
    
    return state
