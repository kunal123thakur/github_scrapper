"""
LeetCode search agent
"""
import numpy as np
import faiss
from database.vector_store import vector_store
from models.state import AgentState


def leetcode_search_agent(state: AgentState) -> AgentState:
    """Searches general LeetCode questions"""
    intent = state["intent_classification"]
    query = state["user_message"]
    
    search_query = f"{query} {' '.join(intent.get('topics', []))}"
    num_questions = int(intent.get("num_questions", 15))
    difficulty = (intent.get("difficulty") or "any").lower()
    
    print(f"🔍 LeetCode: {num_questions} questions, difficulty={difficulty}")
    
    query_embedding = vector_store.embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    retrieval_multiplier = 5 if difficulty != "any" else 2
    k = min(num_questions * retrieval_multiplier, len(vector_store.leetcode_df))
    similarities, indices = vector_store.leetcode_index.search(query_embedding, k)
    
    matched_questions = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        similarity = float(similarities[0][i])
        
        question = vector_store.leetcode_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "leetcode"
        matched_questions.append(question)
    
    if difficulty != "any":
        matched_questions = [
            q for q in matched_questions 
            if q.get("difficulty", "").lower() == difficulty
        ]
    
    matched_questions = matched_questions[:num_questions]
    
    state["leetcode_questions"] = matched_questions
    state["leetcode_interpretation"] = {
        "search_query": search_query,
        "num_found": len(matched_questions),
        "num_requested": num_questions
    }
    
    print(f"✅ LeetCode: {len(matched_questions)} questions found")
    
    return state
