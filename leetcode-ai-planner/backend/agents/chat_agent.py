"""
Chat agent with robust error handling
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
    """Smart chat agent with guaranteed response"""
    session_id = state["session_id"]
    user_message = state["user_message"]
    
    # Load history
    history = memory_store.get(session_id)
    print(f"📥 Loaded {len(history)} messages from Redis for session '{session_id}'")
    
    if history:
        recent = history[-20:]
        conv_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])
    else:
        conv_context = "No previous conversation."
    
    # ============================================
    # STAGE 1: ROUTING DECISION
    # ============================================
    routing_prompt = f"""
You are a routing agent for Bhindi AI.

CONVERSATION HISTORY:
{conv_context}

USER'S MESSAGE:
{user_message}

ROUTING RULES:
- Route to "specialist" → DSA learning questions (explain, what is, how does, teach me)
- Route to "planner" → Practice requests (give me questions, create schedule, similar questions)
- Route to "chat" → Greetings, personal questions, clarifications, thanks

Output JSON:
{{
  "route": "specialist|planner|chat",
  "reasoning": "brief explanation"
}}
"""
    
    try:
        route_response = model.generate_content(routing_prompt)
        route_text = re.sub(r'```(?:json)?', '', route_response.text).strip()
        route_json = re.search(r'\{.*\}', route_text, re.DOTALL)
        route_decision = json.loads(route_json.group() if route_json else route_text)
        
        route = route_decision.get("route", "chat")
        reasoning = route_decision.get("reasoning", "")
        
    except Exception as e:
        print(f"⚠️  Routing error: {e}")
        route = "chat"
        reasoning = "Error - defaulting to chat"
    
    print(f"🔀 Routing: {route} | {reasoning}")
    
    # ============================================
    # STAGE 2: GENERATE RESPONSE (if staying in chat)
    # ============================================
    chat_response = ""
    
    if route == "chat":
        response_prompt = f"""
You are Bhindi AI with perfect memory.

CONVERSATION HISTORY:
{conv_context}

USER'S MESSAGE:
{user_message}

TASK: Answer warmly and personally. Use conversation history to answer questions about what was discussed before.

Output ONLY the response text (no JSON, no code blocks, just your answer):
"""
        
        try:
            response_obj = model.generate_content(response_prompt)
            chat_response = response_obj.text.strip() if response_obj.text else ""
            
            # Clean up
            chat_response = re.sub(r'```.*?```', '', chat_response, flags=re.DOTALL).strip()
            
            # Ensure we have a response
            if not chat_response:
                chat_response = "Hello! I'm Bhindi AI. I can help you learn DSA concepts or find practice questions. What would you like to do?"
            
        except Exception as e:
            print(f"⚠️  Response generation error: {e}")
            chat_response = "Hello! How can I help you with DSA today?"
        
        # Save to memory
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": chat_response})
        memory_store.set(session_id, history)
        
        # Update state
        state["chat_response"] = chat_response
        state["chat_output"] = chat_response
        state["reroute_to_planner"] = False
        state["route_to_specialist"] = False
        
    elif route == "specialist":
        state["route_to_specialist"] = True
        state["reroute_to_planner"] = False
        state["chat_response"] = ""
        chat_response = ""
        
    elif route == "planner":
        state["reroute_to_planner"] = True
        state["route_to_specialist"] = False
        state["chat_response"] = ""
        chat_response = ""
    
    state["chat_history"] = history
    
    # Safe printing
    print(f"✅ Chat agent complete:")
    print(f"   Route: {route}")
    if route == "chat" and chat_response:
        # Safe slicing with proper check
        preview = chat_response[:100] if len(chat_response) > 100 else chat_response
        print(f"   Response: '{preview}...'")
    
    return state
