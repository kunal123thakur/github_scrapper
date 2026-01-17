"""
BHINDI AI - Advanced Multi-Agent DSA Planner v3.0
Chat-First Architecture with Redis Memory
"""

import os
import re
import json
import pickle
import redis
from datetime import date, timedelta
from typing import TypedDict, Annotated, Sequence, Literal, List, Dict, Optional
import operator

import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
LEETCODE_DATASET_PATH = os.path.join(BASE_DIR, "leetcode-ai-planner", "backend", "data", "leetcode_dataset.csv")
COMPANY_DATASET_PATH = os.path.join(BASE_DIR, "leetcode-ai-planner", "backend", "data", "COMBINED_DSA_All_Companies.xlsx")

# Vector index paths
LEETCODE_VECTOR_INDEX = "leetcode_faiss_index.bin"
LEETCODE_METADATA = "leetcode_metadata.pkl"
COMPANY_VECTOR_INDEX = "company_faiss_index.bin"
COMPANY_METADATA = "company_metadata.pkl"

# ==========================================
# REDIS MEMORY STORE
# ==========================================
# ==========================================
# REDIS MEMORY STORE (REDIS CLOUD)
# ==========================================
class RedisChatMemoryStore:
    """Redis-backed chat memory for conversation history"""
    
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host='redis-13657.c80.us-east-1-2.ec2.cloud.redislabs.com',
                port=13657,
                decode_responses=True,
                username="default",
                password="qDFYNsEVRSi6t2Z8CwhUdiG2JJDtTH3V",
            )
            self.redis.ping()
            print("✅ Redis Cloud connected successfully")
        except Exception as e:
            print(f"❌ Redis connection FAILED: {e}")
            print("⚠️  Running WITHOUT memory - conversations won't persist!")
            self.redis = None
    
    def _key(self, session_id: str) -> str:
        return f"bhindi:chat:session:{session_id}"
    
    def get(self, session_id: str) -> List[Dict[str, str]]:
        if not self.redis:
            print(f"⚠️  Redis not available, returning empty history")
            return []
        
        try:
            data = self.redis.get(self._key(session_id))
            history = json.loads(data) if data else []
            print(f"📥 Loaded {len(history)} messages from Redis for session '{session_id}'")
            if history:
                print(f"   Last message: {history[-1]['content'][:50]}...")
            return history
        except Exception as e:
            print(f"❌ Error loading from Redis: {e}")
            return []
    
    def set(self, session_id: str, history: List[Dict[str, str]]) -> None:
        if not self.redis:
            print(f"⚠️  Redis not available, cannot save history")
            return
        
        try:
            self.redis.set(
                self._key(session_id),
                json.dumps(history),
                ex=60 * 60 * 24  # TTL = 24 hours
            )
            print(f"💾 Saved {len(history)} messages to Redis for session '{session_id}'")
        except Exception as e:
            print(f"❌ Error saving to Redis: {e}")
    
    def clear(self, session_id: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.delete(self._key(session_id))
            print(f"🗑️  Cleared history for session '{session_id}'")
        except Exception as e:
            print(f"❌ Error clearing Redis: {e}")



# Initialize memory store
memory_store = RedisChatMemoryStore()

# ==========================================
# VECTOR DATABASE BUILDER
# ==========================================
def build_leetcode_vector_db():
    """Build vector database for general LeetCode questions"""
    print("📦 Building LeetCode vector database...")
    df = pd.read_csv(LEETCODE_DATASET_PATH)
    
    df["tags"] = df["tags"].apply(
        lambda x: [t.lower().strip() for t in re.findall(r"'([^']*)'", str(x))]
    )
    df["difficulty"] = df["difficulty"].str.lower()
    df["problem_description"] = df["problem_description"].fillna("")
    
    semantic_texts = []
    for idx, row in df.iterrows():
        tags_str = ", ".join(row["tags"])
        semantic_text = f"""
        Problem: {row["task_id"].replace('-', ' ')}
        Tags: {tags_str}
        Difficulty: {row["difficulty"]}
        Description: {row["problem_description"][:500]}
        """.strip()
        semantic_texts.append(semantic_text)
    
    embeddings = embedding_model.encode(semantic_texts, show_progress_bar=True, batch_size=32)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, LEETCODE_VECTOR_INDEX)
    
    metadata = {
        "semantic_texts": semantic_texts,
        "dataframe": df.to_dict('records')
    }
    with open(LEETCODE_METADATA, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ LeetCode DB built: {len(df)} questions")
    return index, metadata

def build_company_vector_db():
    """Build vector database for company-specific questions"""
    print("📦 Building Company-Specific vector database...")
    
    df = pd.read_excel(COMPANY_DATASET_PATH, engine='openpyxl')
    
    df["Difficulty"] = df["Difficulty"].str.lower()
    df["Company"] = df["Company"].str.lower()
    df["Question_Name"] = df["Question_Name"].fillna("")
    
    semantic_texts = []
    for idx, row in df.iterrows():
        semantic_text = f"""
        Problem: {row["Question_Name"]}
        Company: {row["Company"]}
        Difficulty: {row["Difficulty"]}
        Type: {row.get("Question_Type", "Algorithm")}
        Time: {row.get("Time_to_Solve", "")}
        """.strip()
        semantic_texts.append(semantic_text)
    
    embeddings = embedding_model.encode(semantic_texts, show_progress_bar=True, batch_size=32)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, COMPANY_VECTOR_INDEX)
    
    metadata = {
        "semantic_texts": semantic_texts,
        "dataframe": df.to_dict('records')
    }
    with open(COMPANY_METADATA, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Company DB built: {len(df)} questions")
    return index, metadata

# ==========================================
# LOAD OR BUILD DATABASES
# ==========================================
if os.path.exists(LEETCODE_VECTOR_INDEX) and os.path.exists(LEETCODE_METADATA):
    print("📂 Loading LeetCode vector database...")
    leetcode_index = faiss.read_index(LEETCODE_VECTOR_INDEX)
    with open(LEETCODE_METADATA, 'rb') as f:
        leetcode_metadata = pickle.load(f)
    leetcode_df = pd.DataFrame(leetcode_metadata["dataframe"])
    print(f"✅ LeetCode: {len(leetcode_df)} questions loaded")
else:
    leetcode_index, leetcode_metadata = build_leetcode_vector_db()
    leetcode_df = pd.DataFrame(leetcode_metadata["dataframe"])

if os.path.exists(COMPANY_VECTOR_INDEX) and os.path.exists(COMPANY_METADATA):
    print("📂 Loading Company vector database...")
    company_index = faiss.read_index(COMPANY_VECTOR_INDEX)
    with open(COMPANY_METADATA, 'rb') as f:
        company_metadata = pickle.load(f)
    company_df = pd.DataFrame(company_metadata["dataframe"])
    print(f"✅ Company: {len(company_df)} questions loaded")
else:
    company_index, company_metadata = build_company_vector_db()
    company_df = pd.DataFrame(company_metadata["dataframe"])

# ==========================================
# STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    user_message: str
    session_id: str
    chat_history: List[Dict[str, str]]
    
    # Chat agent outputs
    chat_response: Optional[str]
    chat_output: Optional[str]
    reroute_to_planner: bool
    
    # Planner outputs
    intent_classification: dict
    plan: List[str]
    tools_selected: List[str]  # For frontend button highlighting
    
    # LeetCode results
    leetcode_interpretation: dict
    leetcode_questions: list
    
    # Company results
    company_interpretation: dict
    company_questions: list
    
    # Combined results
    combined_schedule: dict
    
    errors: Annotated[Sequence[str], operator.add]
    final_response: dict

# ==========================================
# AGENT 1: CHAT AGENT (ENTRY POINT)
# ==========================================
def chat_agent(state: AgentState) -> AgentState:
    """
    Smart Chat Agent with Proper Context Memory
    """
    session_id = state["session_id"]
    user_message = state["user_message"]
    
    # Load chat history from Redis
    history = memory_store.get(session_id)
    
    # Format history for better context understanding
    if history:
        conversation_context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in history[-10:]  # Last 5 conversation turns
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

Examples of good context handling:
- USER: "easy"
  YOU: "Got it, easy questions. Which topic? (arrays, trees, graphs, etc.)"
  
- USER (next): "arrays"  
  YOU: "Perfect! Easy array questions. How many would you like?"
  
- USER (next): "give me 100"
  YOU: [reroute_to_planner = true, because you have: topic=arrays, difficulty=easy, count=100]

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

If information is incomplete, ask ONE specific clarifying question at a time.

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

Think step by step:
1. What did the user tell me before?
2. What does this new message add?
3. Do I have enough to execute (topic/company + optional count/difficulty)?
4. If not, what ONE thing should I ask?

Output ONLY valid JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        # Extract JSON (handle thinking model output)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            output = json.loads(json_match.group())
        else:
            output = json.loads(text)
        
        chat_response = output.get("response", "I'm here to help with DSA prep!")
        reroute = output.get("reroute_to_planner", False)
        
    except Exception as e:
        print(f"⚠️ Chat agent error: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'N/A'}")
        chat_response = "I'm here to help! Could you clarify what questions you're looking for?"
        reroute = False
    
    # Save to memory (append both user message and assistant response)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": chat_response})
    memory_store.set(session_id, history)
    
    # Update state
    state["chat_response"] = chat_response
    state["chat_output"] = chat_response
    state["reroute_to_planner"] = reroute
    state["chat_history"] = history
    
    print(f"💬 Chat Agent:")
    print(f"   History: {len(history)} messages")
    print(f"   Reroute: {reroute}")
    print(f"   Response: {chat_response[:100]}...")
    
    return state

# ==========================================
# AGENT 2: PLANNER (INTENT CLASSIFIER + TOOL SELECTOR)
# ==========================================
def planner_agent(state: AgentState) -> AgentState:
    """
    Planner that understands conversation history
    """
    user_message = state["user_message"]
    chat_output = state.get("chat_output", "")
    history = state.get("chat_history", [])
    
    # Extract all user messages from history for full context
    all_user_messages = [
        msg['content'] for msg in history 
        if msg['role'] == 'user'
    ]
    
    # Combine all user inputs
    full_user_context = " | ".join(all_user_messages[-5:])  # Last 5 user messages
    
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
- Topics mentioned: arrays, trees, graphs, DP, etc.
- Difficulty: easy, medium, hard
- Count: any number mentioned ("100", "10", "fifty")
- Company: google, uber, adobe, or "all companies"

Intent Classification:
- If user mentions company → "company"
- If user mentions general topics (arrays, trees, etc.) → "leetcode"  
- If both → "hybrid"

Examples:
User said: "easy" | "arrays" | "100 questions"
→ intent_type: "leetcode", topics: ["arrays"], difficulty: "easy", num_questions: 100

User said: "google" | "hard" | "10"
→ intent_type: "company", company_name: "google", difficulty: "hard", num_questions: 10

User said: "all companies" | "give me 100"
→ intent_type: "company", company_name: "all", num_questions: 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "intent_type": "leetcode|company|hybrid",
  "confidence": 0.95,
  "reasoning": "extracted from conversation: topic=X, difficulty=Y, count=Z",
  "company_name": "google|uber|all|..." (if company-related),
  "num_questions": 100,
  "difficulty": "easy|medium|hard|any",
  "duration_days": 21,
  "topics": ["arrays", "trees"],
  "tools_selected": ["leetcode_tool"] or ["company_tool"] or both
}}

Output ONLY JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            intent = json.loads(json_match.group())
        else:
            intent = json.loads(text)
        
        # Ensure defaults
        intent.setdefault("intent_type", "leetcode")
        intent.setdefault("num_questions", 15)
        intent.setdefault("duration_days", 21)
        intent.setdefault("difficulty", "any")
        intent.setdefault("topics", [])
        intent.setdefault("tools_selected", ["leetcode_tool"])
        
        # Force int conversion
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
    
    print(f"🎯 Planner extracted from full conversation:")
    print(f"   Intent: {intent['intent_type']}")
    print(f"   Topics: {intent.get('topics', [])}")
    print(f"   Difficulty: {intent.get('difficulty')}")
    print(f"   Count: {intent['num_questions']}")
    print(f"   Company: {intent.get('company_name', 'N/A')}")
    
    return state


# ==========================================
# AGENT 3: LEETCODE SEARCH AGENT
# ==========================================
def leetcode_search_agent(state: AgentState) -> AgentState:
    """Searches general LeetCode questions"""
    intent = state["intent_classification"]
    query = state["user_message"]
    
    search_query = f"{query} {' '.join(intent.get('topics', []))}"
    num_questions = int(intent.get("num_questions", 15))
    difficulty = (intent.get("difficulty") or "any").lower()
    
    print(f"🔍 LeetCode search: '{search_query}' ({num_questions} questions, difficulty={difficulty})")
    
    query_embedding = embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    retrieval_multiplier = 5 if difficulty != "any" else 2
    k = min(num_questions * retrieval_multiplier, len(leetcode_df))
    similarities, indices = leetcode_index.search(query_embedding, k)
    
    print(f"   📊 Retrieving {k} candidates...")
    
    matched_questions = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        similarity = float(similarities[0][i])
        
        question = leetcode_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "leetcode"
        matched_questions.append(question)
    
    if difficulty != "any":
        before_filter = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if q.get("difficulty", "").lower() == difficulty
        ]
        print(f"   🔍 Difficulty filter: {before_filter} → {len(matched_questions)}")
    
    matched_questions = matched_questions[:num_questions]
    
    state["leetcode_questions"] = matched_questions
    state["leetcode_interpretation"] = {
        "search_query": search_query,
        "num_found": len(matched_questions),
        "num_requested": num_questions
    }
    
    print(f"✅ LeetCode: {len(matched_questions)} questions")
    
    return state

# ==========================================
# AGENT 4: COMPANY SEARCH AGENT
# ==========================================
def company_search_agent(state: AgentState) -> AgentState:
    """Searches company-specific questions"""
    intent = state["intent_classification"]
    query = state["user_message"]
    
    company_name = (intent.get("company_name") or "").lower()
    num_questions = int(intent.get("num_questions", 15))
    difficulty = (intent.get("difficulty") or "any").lower()
    
    if company_name == "all" or company_name == "":
        search_query = f"{query} company interview questions"
        company_name = "all"
        print(f"🏢 Company search: ALL COMPANIES ({num_questions} questions)")
    else:
        search_query = f"{query} {company_name} interview"
        print(f"🏢 Company search: {company_name.upper()} ({num_questions} questions)")
    
    query_embedding = embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    if company_name != "all" and difficulty != "any":
        retrieval_multiplier = 10
    elif company_name != "all" or difficulty != "any":
        retrieval_multiplier = 5
    else:
        retrieval_multiplier = 2
    
    k = min(num_questions * retrieval_multiplier, len(company_df))
    similarities, indices = company_index.search(query_embedding, k)
    
    print(f"   📊 Retrieving {k} candidates...")
    
    matched_questions = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        similarity = float(similarities[0][i])
        
        question = company_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "company"
        matched_questions.append(question)
    
    if company_name and company_name != "all":
        before = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if company_name in q.get("Company", "").lower()
        ]
        print(f"   🔍 Company filter: {before} → {len(matched_questions)}")
    
    if difficulty != "any":
        before = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if q.get("Difficulty", "").lower() == difficulty
        ]
        print(f"   🔍 Difficulty filter: {before} → {len(matched_questions)}")
    
    matched_questions = matched_questions[:num_questions]
    
    state["company_questions"] = matched_questions
    state["company_interpretation"] = {
        "search_query": search_query,
        "company": company_name,
        "num_found": len(matched_questions),
        "num_requested": num_questions
    }
    
    print(f"✅ Company: {len(matched_questions)} questions")
    
    return state

# ==========================================
# AGENT 5: SCHEDULER AGENT
# ==========================================
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

# ==========================================
# ROUTING LOGIC
# ==========================================
def route_from_chat(state: AgentState) -> Literal["planner", "end"]:
    """Route from chat: either to planner or end"""
    if state.get("reroute_to_planner"):
        print("🔀 Chat → Planner")
        return "planner"
    print("🔀 Chat → End")
    return "end"

def route_from_planner(state: AgentState) -> Literal["leetcode", "company", "hybrid"]:
    """Route from planner to appropriate agents"""
    intent_type = state["intent_classification"].get("intent_type", "leetcode")
    print(f"🔀 Planner → {intent_type}")
    return intent_type

# ==========================================
# BUILD LANGGRAPH
# ==========================================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chat", chat_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("leetcode", leetcode_search_agent)
workflow.add_node("company", company_search_agent)
workflow.add_node("scheduler", scheduler_agent)

# Entry point: ALWAYS chat first
workflow.set_entry_point("chat")

# FIX: Chat decides: planner or end (use route_from_chat, not route_from_planner!)
workflow.add_conditional_edges(
    "chat",
    route_from_chat,  # ✅ CORRECT FUNCTION
    {
        "planner": "planner",
        "end": END
    }
)

# Planner routes to appropriate agent
workflow.add_conditional_edges(
    "planner",
    route_from_planner,
    {
        "leetcode": "leetcode",
        "company": "company",
        "hybrid": "leetcode"
    }
)

# LeetCode → scheduler
workflow.add_edge("leetcode", "scheduler")

# Company → scheduler
workflow.add_edge("company", "scheduler")

# Hybrid: scheduler → company → scheduler → end
workflow.add_conditional_edges(
    "scheduler",
    lambda state: "company" if state["intent_classification"].get("intent_type") == "hybrid" and not state.get("company_questions") else "end",
    {
        "company": "company",
        "end": END
    }
)

app_graph = workflow.compile()


# ==========================================
# FASTAPI APPLICATION
# ==========================================
app = FastAPI(title="Bhindi AI v3.0 - Chat-First Architecture")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
def chat(req: ChatRequest):
    """Main endpoint with chat-first flow"""
    initial_state = {
        "user_message": req.message,
        "session_id": req.session_id,
        "errors": [],
        "reroute_to_planner": False
    }
    
    try:
        result = app_graph.invoke(initial_state)
        
        # If chat handled it
        if not result.get("reroute_to_planner"):
            return {
                "success": True,
                "type": "chat",
                "response": result.get("chat_response"),
                "session_id": req.session_id
            }
        
        # If planner executed
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
        "leetcode_questions": len(leetcode_df),
        "company_questions": len(company_df),
        "version": "3.0"
    }

@app.delete("/clear/{session_id}")
def clear_memory(session_id: str):
    """Clear chat history for a session"""
    memory_store.clear(session_id)
    return {"success": True, "message": f"Cleared history for {session_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
