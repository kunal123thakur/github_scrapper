"""
Planner agent with FULL conversation context awareness
"""
import re
import json
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME
from models.state import AgentState

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def planner_agent(state: AgentState) -> AgentState:
    """Context-aware planner that understands previous conversation"""
    user_message = state["user_message"]
    chat_output = state.get("chat_output", "")
    history = state.get("chat_history", [])
    
    # Extract ALL user and assistant messages for full context
    if history:
        # Get last 10 exchanges for context
        recent_history = history[-20:]  # Last 10 full exchanges
        
        conversation_summary = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent_history
        ])
        
        # Extract topics/concepts mentioned in recent conversation
        topics_mentioned = []
        for msg in recent_history:
            content_lower = msg['content'].lower()
            # Look for DSA concepts
            if any(keyword in content_lower for keyword in ['linked list', 'middle', 'two pointer', 'fast slow']):
                topics_mentioned.append("linked list")
            if any(keyword in content_lower for keyword in ['array', 'arrays']):
                topics_mentioned.append("arrays")
            if any(keyword in content_lower for keyword in ['tree', 'binary tree', 'bst']):
                topics_mentioned.append("trees")
            if any(keyword in content_lower for keyword in ['graph', 'dfs', 'bfs']):
                topics_mentioned.append("graphs")
            if any(keyword in content_lower for keyword in ['dynamic programming', 'dp', 'memoization']):
                topics_mentioned.append("dynamic programming")
            if any(keyword in content_lower for keyword in ['stack', 'queue']):
                topics_mentioned.append("stack queue")
        
        topics_mentioned = list(set(topics_mentioned))  # Remove duplicates
        
        context_summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL CONVERSATION CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conversation_summary}

TOPICS DETECTED IN RECENT CONVERSATION:
{', '.join(topics_mentioned) if topics_mentioned else 'None detected'}

CRITICAL: The user just had a conversation about these topics. When they ask for "similar questions" or "practice questions", they mean questions related to what was JUST discussed!
"""
    else:
        context_summary = "No previous conversation context."
        topics_mentioned = []
    
    # Combine all user inputs
    all_user_messages = [msg['content'] for msg in history if msg['role'] == 'user']
    full_user_context = " | ".join(all_user_messages[-5:]) if all_user_messages else user_message
    
    prompt = f"""
You are the PLANNER agent with PERFECT MEMORY of the conversation.

{context_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S LATEST REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **READ THE CONVERSATION ABOVE CAREFULLY**
2. If user says "give me similar question" or "practice question" or "can you give", they mean:
   - Questions related to the TOPIC JUST DISCUSSED
   - Same difficulty level as what was discussed
   - Same concepts/patterns

3. Extract parameters from ENTIRE conversation:
   - Topics: What was discussed? (linked list, arrays, trees, etc.)
   - Difficulty: Was it easy/medium/hard? Default: medium
   - Count: How many? Look for numbers. Default: 5
   - Company: Any company mentioned? Default: all

4. When user asks for "similar" questions:
   - Use the topics from the conversation above
   - Maintain same difficulty level
   - Small quantity (1-5 questions)

Examples:
- After discussing "middle of linked list" → user says "give similar question"
  Extract: topics=["linked list", "two pointer"], difficulty="medium", num=1

- User: "give me 10 hard graph questions"
  Extract: topics=["graphs"], difficulty="hard", num=10

- User: "google questions on arrays"
  Extract: topics=["arrays"], company_name="google", difficulty="any", num=15

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "intent_type": "leetcode|company|hybrid",
  "confidence": 0.95,
  "reasoning": "Extracted from conversation: user discussed [TOPIC], now wants similar questions",
  "company_name": "google|all|...",
  "num_questions": 5,
  "difficulty": "easy|medium|hard|any",
  "duration_days": 21,
  "topics": ["linked list", "two pointer"],
  "tools_selected": ["leetcode_tool"]
}}

REMEMBER: "similar question" means use the topics from the conversation!
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        intent = json.loads(json_match.group() if json_match else text)
        
        # Ensure defaults
        intent.setdefault("intent_type", "leetcode")
        intent.setdefault("num_questions", 5)
        intent.setdefault("duration_days", 21)
        intent.setdefault("difficulty", "any")
        intent.setdefault("topics", topics_mentioned if topics_mentioned else [])
        intent.setdefault("tools_selected", ["leetcode_tool"])
        
        # Force int conversion
        intent["num_questions"] = int(intent["num_questions"])
        intent["duration_days"] = int(intent["duration_days"])
        
    except Exception as e:
        print(f"⚠️ Planner error: {e}")
        # Use detected topics as fallback
        intent = {
            "intent_type": "leetcode",
            "num_questions": 5,
            "duration_days": 21,
            "difficulty": "medium",
            "topics": topics_mentioned if topics_mentioned else [],
            "tools_selected": ["leetcode_tool"]
        }
    
    state["intent_classification"] = intent
    state["tools_selected"] = intent["tools_selected"]
    
    print(f"🎯 Planner with context awareness:")
    print(f"   Intent: {intent['intent_type']}")
    print(f"   Topics detected from conversation: {topics_mentioned}")
    print(f"   Topics for search: {intent.get('topics', [])}")
    print(f"   Difficulty: {intent.get('difficulty')}")
    print(f"   Count: {intent['num_questions']}")
    print(f"   Reasoning: {intent.get('reasoning', 'N/A')}")
    

    # ... your existing planner code ...

    state["intent_classification"] = intent
    state["tools_selected"] = intent["tools_selected"]
    
    # 🆕 NEW: Save context for next query (just 3 lines!)
    state["context_topics"] = intent.get("topics", topics_mentioned)
    state["context_company"] = intent.get("company_name", "")
    
    print(f"🎯 Planner with context awareness:")
    print(f"   Intent: {intent['intent_type']}")
    print(f"   Topics detected from conversation: {topics_mentioned}")
    print(f"   Topics for search: {intent.get('topics', [])}")
    print(f"   Difficulty: {intent.get('difficulty')}")
    print(f"   Count: {intent['num_questions']}")
    print(f"   Reasoning: {intent.get('reasoning', 'N/A')}")
    print(f"   💾 Context saved: topics={state['context_topics']}, company={state['context_company']}")  # 🆕 NEW
    
    return state

    return state
