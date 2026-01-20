"""
DSA Specialist with robust JSON handling
"""
import re
import json
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME
from database.dsa_knowledge_store import dsa_knowledge_store
from database.redis_store import memory_store
from models.state import AgentState

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def clean_json_string(text: str) -> str:
    """Clean JSON string by properly escaping control characters"""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?', '', text).strip()
    
    # Find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return text
    
    json_str = json_match.group()
    
    # Escape unescaped newlines inside string values
    # This regex finds strings and replaces \n with \\n
    def escape_newlines(match):
        string_content = match.group(1)
        # Replace literal newlines with escaped newlines
        escaped = string_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return f'"{escaped}"'
    
    # Match quoted strings and escape their content
    cleaned = re.sub(r'"((?:[^"\\]|\\.)*)"', escape_newlines, json_str)
    
    return cleaned


def dsa_specialist_agent(state: AgentState) -> AgentState:
    """DSA Professor with robust JSON parsing"""
    session_id = state["session_id"]
    user_message = state["user_message"]
    
    history = memory_store.get(session_id)
    print(f"📥 DSA Specialist: Loaded {len(history)} messages")
    
    # Retrieve knowledge
    if dsa_knowledge_store:
        knowledge_results = dsa_knowledge_store.search(user_message, top_k=3)
        context = "\n".join([f"[{r['source']}]\n{r['content'][:400]}" for r in knowledge_results])
    else:
        context = "No knowledge available."
    
    if history:
        conv_context = "\n".join([f"{m['role'].upper()}: {m['content'][:100]}" for m in history[-6:]])
    else:
        conv_context = "First interaction."
    
    prompt = f"""
You are Professor DSA.

KNOWLEDGE:
{context}

HISTORY:
{conv_context}

QUESTION:
{user_message}

Provide a clear DSA explanation. Use markdown for formatting.

**CRITICAL JSON FORMAT RULES:**
1. The "response" field must have ALL newlines as \\n (two characters)
2. Use \\n\\n for paragraph breaks
3. Use \\n for line breaks
4. No literal newlines inside the JSON string

Example correct format:
{{
  "response": "First paragraph.\\n\\nSecond paragraph with\\nline break.",
  "concepts_covered": ["arrays"],
  "complexity": {{"time": "O(n)"}},
  "follow_up": "",
  "confidence": "high"
}}

Output valid JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        
        # Clean and parse
        cleaned_text = clean_json_string(response.text)
        
        try:
            output = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Cleaned text: {cleaned_text[:300]}")
            # Fallback: extract response manually
            output = {"response": response.text[:500], "concepts_covered": [], "confidence": "low"}
        
        specialist_response = output.get("response", "I can help with DSA questions!")
        concepts = output.get("concepts_covered", [])
        complexity = output.get("complexity", {})
        follow_up = output.get("follow_up", "")
        confidence = output.get("confidence", "medium")
        
        # Convert \\n to actual newlines for display
        specialist_response = specialist_response.replace('\\n', '\n')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        specialist_response = "Could you rephrase your DSA question?"
        concepts = []
        complexity = {}
        follow_up = ""
        confidence = "low"
    
    # Save
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": specialist_response})
    memory_store.set(session_id, history)
    
    # Update state
    state["chat_response"] = specialist_response
    state["chat_output"] = specialist_response
    state["dsa_concepts_covered"] = concepts
    state["dsa_complexity"] = complexity
    state["dsa_follow_up"] = follow_up
    state["dsa_confidence"] = confidence
    state["reroute_to_planner"] = False
    state["chat_history"] = history
    
    print(f"✅ DSA Response: {len(specialist_response)} chars, {len(concepts)} concepts")
    
    return state
