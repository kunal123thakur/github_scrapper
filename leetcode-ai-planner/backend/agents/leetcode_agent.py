"""
LeetCode search agent with proper FAISS indexing
"""
import numpy as np
import faiss
from database.vector_store import vector_store
from models.state import AgentState


def leetcode_search_agent(state: AgentState) -> AgentState:
    """Searches LeetCode questions with context-aware query"""
    intent = state["intent_classification"]
    user_message = state["user_message"]
    
    # Build search query from topics
    topics = intent.get('topics', [])
    difficulty = (intent.get("difficulty") or "any").lower()
    num_questions = int(intent.get("num_questions", 5))
    
    # Create rich search query
    if topics:
        topic_query = " ".join(topics)
        search_query = f"{topic_query} algorithm problem solving"
    else:
        search_query = user_message
    
    print(f"🔍 LeetCode search:")
    print(f"   Query: '{search_query}'")
    print(f"   Topics: {topics}")
    print(f"   Difficulty: {difficulty}")
    print(f"   Count: {num_questions}")
    
    # Create embedding
    query_embedding = vector_store.embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    
    # Retrieve candidates
    retrieval_multiplier = 10 if difficulty != "any" else 5
    k = min(num_questions * retrieval_multiplier, len(vector_store.leetcode_df))
    
    # FAISS search returns (similarities, indices) as 2D arrays
    similarities, indices = vector_store.leetcode_index.search(query_embedding, k)
    
    print(f"   📊 Retrieving {k} candidates...")
    
    matched_questions = []
    
    # FIX: indices and similarities are 2D arrays [batch_size, k]
    # Since batch_size=1, we access [0][i]
    for i in range(len(indices[0])):
        idx = int(indices[0][i])  # ✅ CORRECT: indices[0][i]
        similarity = float(similarities[0][i])  # ✅ CORRECT: similarities[0][i]
        
        # Get question from dataframe
        question = vector_store.leetcode_df.iloc[idx].to_dict()
        question["similarity_score"] = similarity
        question["source"] = "leetcode"
        
        # Boost similarity if tags match topics
        if topics:
            question_tags = [tag.lower() for tag in question.get("tags", [])]
            for topic in topics:
                # Check if topic appears in any tag
                if any(topic.lower() in tag for tag in question_tags):
                    question["similarity_score"] += 0.1  # Boost relevance
        
        matched_questions.append(question)
    
    # Sort by boosted similarity
    matched_questions = sorted(matched_questions, key=lambda x: -x["similarity_score"])
    
    # Filter by difficulty
    if difficulty != "any":
        before = len(matched_questions)
        matched_questions = [
            q for q in matched_questions 
            if q.get("difficulty", "").lower() == difficulty
        ]
        print(f"   🔍 Difficulty filter: {before} → {len(matched_questions)}")
    
    # Take top N
    matched_questions = matched_questions[:num_questions]
    
    # Calculate average similarity for debugging
    avg_sim = np.mean([q['similarity_score'] for q in matched_questions]) if matched_questions else 0
    
    state["leetcode_questions"] = matched_questions
    state["leetcode_interpretation"] = {
        "search_query": search_query,
        "topics_used": topics,
        "num_found": len(matched_questions),
        "num_requested": num_questions
    }
    
    print(f"✅ LeetCode: {len(matched_questions)} questions")
    print(f"   Average similarity: {avg_sim:.3f}")
    if matched_questions:
        print(f"   Top result: {matched_questions[0].get('task_id', 'N/A')} (sim: {matched_questions[0]['similarity_score']:.3f})")
    
    return state
