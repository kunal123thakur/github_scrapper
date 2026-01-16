import os
import re
import json
import pickle
from datetime import date, timedelta
from typing import TypedDict, Annotated, Sequence, Literal
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
DATASET_PATH = os.path.join(BASE_DIR, "leetcode-ai-planner", "backend", "data", "leetcode_dataset.csv")
VECTOR_INDEX_PATH = "leetcode_faiss_index.bin"
METADATA_PATH = "leetcode_metadata.pkl"

# ==========================================
# BUILD VECTOR DATABASE
# ==========================================
def build_vector_database():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    df["tags"] = df["tags"].apply(
        lambda x: [t.lower().strip() for t in re.findall(r"'([^']*)'", str(x))]
    )
    df["difficulty"] = df["difficulty"].str.lower()
    df["problem_description"] = df["problem_description"].fillna("")
    
    print(f"Loaded {len(df)} questions")
    
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
    
    print("Generating embeddings for all questions...")
    embeddings = embedding_model.encode(
        semantic_texts, 
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, VECTOR_INDEX_PATH)
    
    metadata = {
        "semantic_texts": semantic_texts,
        "dataframe": df.to_dict('records')
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Vector database built! Index saved to {VECTOR_INDEX_PATH}")
    return index, metadata

# ==========================================
# LOAD DATABASE
# ==========================================
if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(METADATA_PATH):
    print("Loading existing vector database...")
    vector_index = faiss.read_index(VECTOR_INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    df_records = metadata["dataframe"]
    df = pd.DataFrame(df_records)
    print(f"✅ Loaded {len(df)} questions from cache")
else:
    print("Building vector database for the first time...")
    vector_index, metadata = build_vector_database()
    df_records = metadata["dataframe"]
    df = pd.DataFrame(df_records)

# ==========================================
# STATE
# ==========================================
class AgentState(TypedDict):
    user_query: str
    interpreted_intent: dict
    retrieved_questions: list
    plan: dict
    filtered_questions: list
    schedule: dict
    errors: Annotated[Sequence[str], operator.add]
    confidence_score: float

# ==========================================
# AGENT 1: INTENT (FIXED)
# ==========================================
def intent_agent(state: AgentState) -> AgentState:
    query = state["user_query"]
    
    prompt = f"""
You are an expert at understanding coding practice requests.

User query: "{query}"

Extract:
1. What topics/concepts? (be descriptive)
2. How many questions? (must be a NUMBER, default 15)
3. Difficulty? (easy/medium/hard/any)
4. Duration? (must be a NUMBER, default 21)

Output JSON (num_questions and duration_days MUST be numbers, not null):
{{
  "search_query": "expanded semantic search",
  "num_questions": 15,
  "difficulty": "any",
  "duration_days": 21,
  "user_intent_summary": "brief summary"
}}

Example:
- "fibonacci questions" → search_query: "fibonacci sequence dynamic programming recursion math"

Output ONLY JSON:
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            intent = json.loads(json_match.group())
        else:
            intent = json.loads(text)
        
        # CRITICAL FIX: Ensure required fields are valid numbers
        if not isinstance(intent.get("num_questions"), (int, float)) or intent.get("num_questions") is None:
            intent["num_questions"] = 15
        
        if not isinstance(intent.get("duration_days"), (int, float)) or intent.get("duration_days") is None:
            intent["duration_days"] = 21
            
        # Convert to int
        intent["num_questions"] = int(intent["num_questions"])
        intent["duration_days"] = int(intent["duration_days"])
        
    except Exception as e:
        print(f"⚠️ Intent parsing error: {e}")
        intent = {
            "search_query": query,
            "num_questions": 15,
            "difficulty": "any",
            "duration_days": 21,
            "user_intent_summary": query
        }
    
    state["interpreted_intent"] = intent
    print(f"📝 Intent: {intent['num_questions']} questions, {intent['duration_days']} days")
    return state
# ==========================================
# AGENT 2: SEMANTIC SEARCH (FINAL FIX)
# ==========================================
def semantic_search_agent(state: AgentState) -> AgentState:
    intent = state["interpreted_intent"]
    search_query = intent.get("search_query", state["user_query"])
    num_questions = int(intent.get("num_questions") or 15)
    difficulty = intent.get("difficulty", "any")
    
    print(f"🔍 Searching for: '{search_query}'")
    
    # Embed query
    query_embedding = embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    k = min(num_questions * 3, len(df))
    similarities, indices = vector_index.search(query_embedding, k)
    
    # CRITICAL FIX: Extract first row and convert numpy types properly
    matched_questions = []
    indices_list = indices[0].tolist()  # Convert to Python list
    similarities_list = similarities[0].tolist()  # Convert to Python list
    
    for idx, similarity in zip(indices_list, similarities_list):
        question = df.iloc[idx].to_dict()  # Now idx is a Python int
        question["similarity_score"] = float(similarity)
        matched_questions.append(question)
    
    # Filter by difficulty
    if difficulty != "any":
        matched_questions = [
            q for q in matched_questions 
            if q.get("difficulty", "").lower() == difficulty.lower()
        ]
    
    # Take top N
    matched_questions = matched_questions[:num_questions]
    
    if matched_questions:
        avg_similarity = float(np.mean([q["similarity_score"] for q in matched_questions]))
    else:
        avg_similarity = 0.0
    
    state["retrieved_questions"] = matched_questions
    state["confidence_score"] = avg_similarity
    
    print(f"✅ Retrieved {len(matched_questions)} questions (avg similarity: {avg_similarity:.3f})")
    
    return state

# ==========================================
# AGENT 3: VALIDATION
# ==========================================
def validation_agent(state: AgentState) -> AgentState:
    if not state.get("retrieved_questions"):
        state["errors"] = state.get("errors", []) + ["No matching questions found"]
    return state

# ==========================================
# AGENT 4: PLANNER (FIXED)
# ==========================================
def planner_agent(state: AgentState) -> AgentState:
    questions = state.get("retrieved_questions", [])
    intent = state.get("interpreted_intent", {})
    duration = int(intent.get("duration_days") or 21)
    
    if not questions:
        state["plan"] = {
            "duration_days": duration,
            "total_questions": 0
        }
        return state
    
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for q in questions:
        diff = q.get("difficulty", "").lower()
        if diff in by_difficulty:
            by_difficulty[diff].append(q)
    
    state["plan"] = {
        "duration_days": duration,
        "total_questions": len(questions),
        "by_difficulty": {k: len(v) for k, v in by_difficulty.items()},
        "average_confidence": state.get("confidence_score", 0.0)
    }
    
    return state

# ==========================================
# AGENT 5: SCHEDULER (FIXED)
# ==========================================
def scheduler_agent(state: AgentState) -> AgentState:
    questions = state.get("retrieved_questions", [])
    plan = state.get("plan", {})
    duration = int(plan.get("duration_days") or 21)
    
    if not questions:
        state["schedule"] = {}
        return state
    
    # Ensure duration is valid
    if duration <= 0:
        duration = 21
    
    questions_sorted = sorted(
        questions, 
        key=lambda x: (-x.get("similarity_score", 0), x.get("difficulty", ""))
    )
    
    start = date.today()
    schedule_map = {}
    
    for i, q in enumerate(questions_sorted):
        day = start + timedelta(days=i % duration)  # duration is guaranteed to be int > 0
        schedule_map.setdefault(str(day), []).append({
            "title": q.get("task_id", "").replace("-", " ").title(),
            "difficulty": q.get("difficulty", "").title(),
            "tags": q.get("tags", [])[:3],
            "url": f"https://leetcode.com/problems/{q.get('task_id', '')}/",
            "similarity": round(q.get("similarity_score", 0), 3),
            "description_preview": q.get("problem_description", "")[:150] + "..."
        })
    
    state["schedule"] = schedule_map
    print(f"📅 Scheduled {len(questions)} questions over {duration} days")
    return state

# ==========================================
# BUILD GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("intent", intent_agent)
workflow.add_node("semantic_search", semantic_search_agent)
workflow.add_node("validate", validation_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("scheduler", scheduler_agent)

workflow.set_entry_point("intent")
workflow.add_edge("intent", "semantic_search")
workflow.add_edge("semantic_search", "validate")
workflow.add_edge("validate", "planner")
workflow.add_edge("planner", "scheduler")
workflow.add_edge("scheduler", END)

app_graph = workflow.compile()

# ==========================================
# FASTAPI
# ==========================================
app = FastAPI(title="Dynamic LeetCode Planner with Semantic Search")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    initial_state = {
        "user_query": req.message,
        "errors": []
    }
    
    try:
        result = app_graph.invoke(initial_state)
        
        return {
            "success": True,
            "query": req.message,
            "interpretation": result.get("interpreted_intent"),
            "confidence": result.get("confidence_score"),
            "plan": result.get("plan"),
            "schedule": result.get("schedule"),
            "total_questions": len(result.get("retrieved_questions", [])),
            "warnings": result.get("errors", []),
            "sample_questions": [
                {
                    "title": q.get("task_id"),
                    "similarity": q.get("similarity_score"),
                    "tags": q.get("tags")
                }
                for q in result.get("retrieved_questions", [])[:5]
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "total_questions": len(df),
        "vector_index_size": vector_index.ntotal
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
