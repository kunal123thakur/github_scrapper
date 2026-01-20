"""
DSA Specialist Agent - RAG-powered DSA Professor
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


def dsa_specialist_agent(state: AgentState) -> AgentState:
    """DSA Professor with RAG capabilities"""
    session_id = state["session_id"]
    user_message = state["user_message"]
    
    # Retrieve relevant knowledge from PDFs
    if dsa_knowledge_store:
        print(f"🔍 Searching DSA knowledge for: '{user_message[:50]}...'")
        knowledge_results = dsa_knowledge_store.search(user_message, top_k=4)
        
        # Format context with sources
        context_parts = []
        for i, result in enumerate(knowledge_results, 1):
            context_parts.append(f"""
[Reference {i} - {result['source']} - Similarity: {result['similarity']:.3f}]
{result['content']}
""")
        context = "\n".join(context_parts)
        
        print(f"✅ Retrieved {len(knowledge_results)} relevant chunks")
    else:
        context = "Knowledge base not available."
        print("⚠️  No knowledge base loaded")
    
    # Load conversation history
    history = memory_store.get(session_id)
    
    if history:
        conversation_context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in history[-8:]  # Last 4 exchanges
        ])
    else:
        conversation_context = "First interaction with this student."
    
    prompt = f"""
You are **Professor DSA**, a world-class expert in Data Structures and Algorithms with decades of teaching experience. You combine deep technical knowledge with an approachable, encouraging teaching style.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 RETRIEVED KNOWLEDGE FROM DSA TEXTBOOKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conversation_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 STUDENT'S QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 YOUR TEACHING GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Ground your answer in the retrieved knowledge above** - Reference specific concepts, algorithms, or techniques from the textbook excerpts
2. **Explain with clarity** - Start with intuition, then provide technical details
3. **Use examples** - Provide concrete examples or code snippets when helpful (Python preferred)
4. **Complexity analysis** - Always mention time/space complexity for algorithms
5. **Build upon previous conversation** - Reference earlier questions if relevant
6. **Encourage learning** - Be warm and supportive, acknowledge good questions
7. **Structure your response**:
   - Brief direct answer first
   - Detailed explanation with examples
   - Complexity analysis (if applicable)
   - Related concepts or next steps

IMPORTANT: If the retrieved knowledge contains relevant information, USE IT and reference it in your explanation. If the question is about concepts not well-covered in the retrieved chunks, use your general DSA knowledge but acknowledge this.

Output format:
{{
  "response": "Your detailed, educational response as Professor DSA. Use markdown for code blocks and formatting.",
  "concepts_covered": ["concept1", "concept2", "..."],
  "complexity_mentioned": {{"time": "O(n)", "space": "O(1)"}},
  "follow_up_suggestion": "Optional: What the student should explore next",
  "confidence": "high/medium/low based on relevance of retrieved knowledge"
}}

Be professorial yet warm. Make DSA accessible and exciting!
"""
    
    try:
        response = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', response.text).strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            output = json.loads(json_match.group())
        else:
            output = json.loads(text)
        
        specialist_response = output.get("response", "I'm here to help you understand DSA!")
        concepts = output.get("concepts_covered", [])
        complexity = output.get("complexity_mentioned", {})
        follow_up = output.get("follow_up_suggestion", "")
        confidence = output.get("confidence", "medium")
        
        print(f"🎓 Professor DSA responded with {confidence} confidence")
        print(f"   Concepts covered: {', '.join(concepts[:3])}")
        
    except Exception as e:
        print(f"⚠️ DSA Specialist error: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'N/A'}")
        specialist_response = "I'm having trouble processing that question. Could you rephrase it or ask about a specific DSA topic?"
        concepts = []
        complexity = {}
        follow_up = ""
        confidence = "low"
    
    # Save conversation
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
    
    return state
