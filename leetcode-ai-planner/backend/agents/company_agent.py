"""
Company-specific search agent with proper FAISS indexing
"""
import numpy as np
import faiss
from database.vector_store import vector_store
from models.state import AgentState


def company_search_agent(state: AgentState) -> AgentState:
    """Searches company-specific questions"""
    intent = state["intent_classification"]
    query = state["user_message"]
    
    company_name = (intent.get("company_name") or "").lower()
    num_questions = int(intent.get("num_questions", 15))
    difficulty = (intent.get("difficulty") or "any").lower()
    
    # Parse multiple companies
    if company_name and company_name not in ["all", ""]:
        companies = [c.strip() for c in company_name.replace(" and ", ",").split(",")]
        companies = [c for c in companies if c]
    else:
        companies = []
    
    if not companies or company_name == "all":
        search_query = f"{query} company interview questions"
        company_name = "all"
        print(f"🏢 Company: ALL ({num_questions} questions)")
    else:
        search_query = f"{query} {' '.join(companies)} interview"
        print(f"🏢 Company: {', '.join([c.upper() for c in companies])} ({num_questions} questions)")
    
    query_embedding = vector_store.embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    if company_name != "all" and difficulty != "any":
        retrieval_multiplier = 10
    elif company_name != "all" or difficulty != "any":
        retrieval_multiplier = 5
    else:
        retrieval_multiplier = 2
    
    k = min(num_questions * retrieval_multiplier, len(vector_store.company_df))
    similarities, indices = vector_store.company_index.search(query_embedding, k)
    
    print(f"   📊 Retrieving {k} candidates...")
    
    matched_questions = []
    
    # FIX: Proper 2D array indexing
    for i in range(len(indices[0])):
        idx = int(indices[0][i])  # ✅ CORRECT
        similarity = float(similarities[0][i])  # ✅ CORRECT
        
        question = vector_store.company_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "company"
        matched_questions.append(question)
    
    # Filter by companies
    if company_name != "all" and companies:
        before = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if any(comp in q.get("Company", "").lower() for comp in companies)
        ]
        print(f"   🔍 Company filter ({', '.join(companies)}): {before} → {len(matched_questions)}")
    
    # Filter by difficulty
    if difficulty != "any":
        before = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if q.get("Difficulty", "").lower() == difficulty
        ]
        print(f"   🔍 Difficulty filter: {before} → {len(matched_questions)}")
    
    matched_questions = matched_questions[:num_questions]
    
    if len(matched_questions) < num_questions:
        print(f"   ⚠️  Only found {len(matched_questions)} out of {num_questions} requested")
    
    state["company_questions"] = matched_questions
    state["company_interpretation"] = {
        "search_query": search_query,
        "company": company_name,
        "companies": companies if companies else ["all"],
        "num_found": len(matched_questions),
        "num_requested": num_questions
    }
    
    print(f"✅ Company: {len(matched_questions)} questions")
    
    return state
