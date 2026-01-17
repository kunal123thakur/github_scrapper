"""
BHINDI AI - Advanced Multi-Agent DSA Planner
Version 2.0 - LeetCode + Company-Specific Questions
"""

import os
import re
import json
import pickle
from datetime import date, timedelta
from typing import TypedDict, Annotated, Sequence, Literal, List, Dict
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
COMPANY_DATASET_PATH = os.path.join(BASE_DIR, "leetcode-ai-planner", "backend", "data", "COMBINED_DSA_All_Companies.xlsx")  # Your new dataset

# Vector index paths
LEETCODE_VECTOR_INDEX = "leetcode_faiss_index.bin"
LEETCODE_METADATA = "leetcode_metadata.pkl"
COMPANY_VECTOR_INDEX = "company_faiss_index.bin"
COMPANY_METADATA = "company_metadata.pkl"

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
    
    # FIX: Use read_excel for .xlsx files
    df = pd.read_excel(COMPANY_DATASET_PATH, engine='openpyxl')
    
    # Normalize columns
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
# LeetCode Database
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

# Company Database
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
    user_query: str
    intent_classification: dict  # {type: "chat"/"leetcode"/"company"/"hybrid", ...}
    
    # LeetCode results
    leetcode_interpretation: dict
    leetcode_questions: list
    leetcode_schedule: dict
    
    # Company results
    company_interpretation: dict
    company_questions: list
    company_schedule: dict
    
    # Combined results
    combined_schedule: dict
    
    # Chat response
    chat_response: str
    
    errors: Annotated[Sequence[str], operator.add]
    final_response: dict

# ==========================================
# AGENT 1: INTENT CLASSIFIER (MASTER ROUTER)
# ==========================================
def intent_classifier_agent(state: AgentState) -> AgentState:
    """
    Classifies user intent into:
    - "chat" → Casual conversation
    - "leetcode" → General DSA questions
    - "company" → Company-specific questions
    - "hybrid" → Both LeetCode + Company
    """
    query = state["user_query"]
    
    prompt = f"""
You are an expert intent classifier for a DSA question planning system.

User query: "{query}"

Classify the intent into ONE of these categories:

1. **chat** - Casual conversation (hello, how are you, what can you do)
2. **leetcode** - General DSA questions (fibonacci, trees, graphs, DP)
3. **company** - Company-specific questions (Google questions, Uber interview prep)
4. **hybrid** - Both general + company-specific (LeetCode problems asked at Google)

Output JSON:
{{
  "intent_type": "chat|leetcode|company|hybrid",
  "confidence": 0.95,
  "reasoning": "brief explanation",
  "company_name": "google|uber|adobe|..." (if company-related),
  "num_questions": 15 (if task-related),
  "difficulty": "easy|medium|hard|any",
  "duration_days": 21,
  "topics": ["arrays", "graphs"] (if specified)
}}

Examples:
- "hello" → intent_type: "chat"
- "fibonacci questions" → intent_type: "leetcode"
- "Google interview questions" → intent_type: "company", company_name: "google"
- "LeetCode hard problems asked at Uber" → intent_type: "hybrid", company_name: "uber"

Output ONLY JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        intent = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group())
        
        # Ensure defaults
        if "intent_type" not in intent:
            intent["intent_type"] = "chat"
        if "num_questions" not in intent or intent["num_questions"] is None:
            intent["num_questions"] = 15
        if "duration_days" not in intent or intent["duration_days"] is None:
            intent["duration_days"] = 21
            
    except Exception as e:
        print(f"⚠️ Intent classification error: {e}")
        intent = {
            "intent_type": "chat",
            "confidence": 0.5,
            "reasoning": "Defaulting to chat",
            "num_questions": 15,
            "duration_days": 21
        }
    
    state["intent_classification"] = intent
    print(f"🎯 Intent: {intent['intent_type']} (confidence: {intent.get('confidence', 0):.2f})")
    
    return state

# ==========================================
# AGENT 2: CHAT AGENT
# ==========================================
def chat_agent(state: AgentState) -> AgentState:
    """Handles casual conversation"""
    query = state["user_query"]
    
    prompt = f"""
You are Bhindi AI, a friendly DSA interview preparation assistant.

User says: "{query}"

Respond naturally and helpfully. Explain what you can do:
- Help plan LeetCode practice schedules
- Find company-specific interview questions (Google, Uber, Adobe, etc.)
- Create personalized study plans
- Search questions by topic, difficulty, company

Keep response concise (2-3 sentences).
"""
    
    try:
        response = model.generate_content(prompt)
        chat_response = response.text.strip()
    except Exception as e:
        chat_response = "Hello! I'm Bhindi AI. I help you prepare for coding interviews with personalized question schedules from LeetCode and company-specific questions!"
    
    state["chat_response"] = chat_response
    return state
# ==========================================
# AGENT 3: LEETCODE SEMANTIC SEARCH AGENT (FIXED)
# ==========================================
def leetcode_search_agent(state: AgentState) -> AgentState:
    """Searches general LeetCode questions using semantic search"""
    intent = state["intent_classification"]
    query = state["user_query"]
    
    search_query = f"{query} {' '.join(intent.get('topics', []))}"
    num_questions = int(intent.get("num_questions", 15))
    difficulty = intent.get("difficulty", "any")
    
    print(f"🔍 LeetCode search: '{search_query}'")
    
    query_embedding = embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    k = min(num_questions * 3, len(leetcode_df))
    similarities, indices = leetcode_index.search(query_embedding, k)
    
    matched_questions = []
    # FIX: FAISS returns 2D arrays, need [0] to get first row
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        similarity = float(similarities[0][i])
        
        question = leetcode_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "leetcode"
        matched_questions.append(question)
    
    if difficulty != "any":
        matched_questions = [q for q in matched_questions if q.get("difficulty", "").lower() == difficulty.lower()]
    
    matched_questions = matched_questions[:num_questions]
    
    state["leetcode_questions"] = matched_questions
    state["leetcode_interpretation"] = {
        "search_query": search_query,
        "num_found": len(matched_questions),
        "avg_similarity": float(np.mean([q["similarity_score"] for q in matched_questions])) if matched_questions else 0.0
    }
    
    print(f"✅ LeetCode: {len(matched_questions)} questions found")
    
    return state

# ==========================================
# AGENT 4: COMPANY-SPECIFIC SEARCH AGENT (FIXED)
# ==========================================
def company_search_agent(state: AgentState) -> AgentState:
    """Searches company-specific questions with intelligent filtering"""
    intent = state["intent_classification"]
    query = state["user_query"]
    
    # FIX: Handle None company_name
    company_name = (intent.get("company_name") or "").lower()
    num_questions = int(intent.get("num_questions", 15))
    difficulty = (intent.get("difficulty") or "any").lower()
    
    # Build search query
    if company_name == "all" or company_name == "":
        search_query = f"{query} company interview questions algorithm"
        company_name = "all"  # Normalize empty to "all"
        print(f"🏢 Company search: ALL COMPANIES - {num_questions} questions (difficulty={difficulty})")
    else:
        search_query = f"{query} {company_name} interview questions"
        print(f"🏢 Company search: {company_name.upper()} - {num_questions} questions (difficulty={difficulty})")
    
    query_embedding = embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    # Smart retrieval multiplier based on filtering needs
    if company_name != "all" and difficulty != "any":
        retrieval_multiplier = 10  # Both filters
    elif company_name != "all" or difficulty != "any":
        retrieval_multiplier = 5   # One filter
    else:
        retrieval_multiplier = 2   # No filters
    
    k = min(num_questions * retrieval_multiplier, len(company_df))
    similarities, indices = company_index.search(query_embedding, k)
    
    print(f"   📊 Retrieving {k} candidates for filtering...")
    
    matched_questions = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        similarity = float(similarities[0][i])
        
        question = company_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "company"
        matched_questions.append(question)
    
    # Apply company filter (skip if "all")
    if company_name and company_name != "all":
        before_company_filter = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if company_name in q.get("Company", "").lower()
        ]
        print(f"   🔍 Company filter: {before_company_filter} → {len(matched_questions)} questions")
    
    # Apply difficulty filter
    if difficulty != "any":
        before_diff_filter = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if q.get("Difficulty", "").lower() == difficulty.lower()
        ]
        print(f"   🔍 Difficulty filter: {before_diff_filter} → {len(matched_questions)} questions")
    
    # Take top N
    matched_questions = matched_questions[:num_questions]
    
    if len(matched_questions) < num_questions:
        print(f"   ⚠️  Only found {len(matched_questions)} out of {num_questions} requested")
    
    state["company_questions"] = matched_questions
    state["company_interpretation"] = {
        "search_query": search_query,
        "company": company_name,
        "num_found": len(matched_questions),
        "num_requested": num_questions,
        "avg_similarity": float(np.mean([q["similarity_score"] for q in matched_questions])) if matched_questions else 0.0
    }
    
    print(f"✅ Company: {len(matched_questions)} questions returned")
    
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
    
    # Collect questions based on intent
    if intent_type in ["leetcode", "hybrid"]:
        all_questions.extend(state.get("leetcode_questions", []))
    
    if intent_type in ["company", "hybrid"]:
        all_questions.extend(state.get("company_questions", []))
    
    if not all_questions:
        state["combined_schedule"] = {}
        return state
    
    # Sort by similarity
    all_questions = sorted(all_questions, key=lambda x: -x.get("similarity_score", 0))
    
    start = date.today()
    schedule_map = {}
    
    for i, q in enumerate(all_questions):
        day = start + timedelta(days=i % duration)
        
        # Format question based on source
        if q.get("source") == "leetcode":
            question_entry = {
                "title": q.get("task_id", "").replace("-", " ").title(),
                "difficulty": q.get("difficulty", "").title(),
                "tags": q.get("tags", [])[:3],
                "url": f"https://leetcode.com/problems/{q.get('task_id', '')}/",
                "source": "LeetCode",
                "similarity": round(q.get("similarity_score", 0), 3)
            }
        else:  # company source
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
def route_by_intent(state: AgentState) -> Literal["chat", "leetcode", "company", "hybrid"]:
    """Routes to appropriate agent based on intent"""
    intent_type = state["intent_classification"].get("intent_type", "chat")
    print(f"🔀 Routing to: {intent_type}")
    return intent_type

# ==========================================
# BUILD LANGGRAPH
# ==========================================
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("intent_classifier", intent_classifier_agent)
workflow.add_node("chat", chat_agent)
workflow.add_node("leetcode", leetcode_search_agent)
workflow.add_node("company", company_search_agent)
workflow.add_node("scheduler", scheduler_agent)

# Set entry point
workflow.set_entry_point("intent_classifier")

# Conditional routing from intent classifier
workflow.add_conditional_edges(
    "intent_classifier",
    route_by_intent,
    {
        "chat": "chat",
        "leetcode": "leetcode",
        "company": "company",
        "hybrid": "leetcode"  # Hybrid goes to leetcode first
    }
)

# Chat path
workflow.add_edge("chat", END)

# LeetCode path
workflow.add_edge("leetcode", "scheduler")

# Company path
workflow.add_edge("company", "scheduler")

# For hybrid: after leetcode, go to company
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
app = FastAPI(title="Bhindi AI - Multi-Agent DSA Planner v2.0")

class ChatRequest(BaseModel):
    message: str
    tool: str = "auto"  # "auto", "leetcode", "company", "both"

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Main endpoint - handles all types of queries
    """
    initial_state = {
        "user_query": req.message,
        "errors": []
    }
    
    try:
        result = app_graph.invoke(initial_state)
        
        intent_type = result["intent_classification"].get("intent_type")
        
        # Format response based on intent
        if intent_type == "chat":
            return {
                "success": True,
                "intent": "chat",
                "response": result.get("chat_response"),
                "query": req.message
            }
        else:
            return {
                "success": True,
                "intent": intent_type,
                "query": req.message,
                "classification": result.get("intent_classification"),
                "leetcode_summary": result.get("leetcode_interpretation"),
                "company_summary": result.get("company_interpretation"),
                "schedule": result.get("combined_schedule"),
                "total_questions": (
                    len(result.get("leetcode_questions", [])) + 
                    len(result.get("company_questions", []))
                ),
                "warnings": result.get("errors", [])
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
        "version": "2.0"
    }

@app.get("/companies")
def list_companies():
    """List all available companies"""
    companies = company_df["Company"].unique().tolist()
    return {
        "companies": companies,
        "count": len(companies)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
