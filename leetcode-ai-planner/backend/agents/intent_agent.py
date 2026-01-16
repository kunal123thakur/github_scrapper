import google.generativeai as genai
import json
import re
from backend.config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

VALID_TAGS = [
    'array', 'backtracking', 'biconnected component', 'binary indexed tree', 'binary search', 'binary search tree',
    'binary tree', 'bit manipulation', 'bitmask', 'brainteaser', 'breadth-first search', 'bucket sort',
    'combinatorics', 'concurrency', 'counting', 'counting sort', 'depth-first search', 'divide and conquer',
    'dynamic programming', 'enumeration', 'eulerian circuit', 'game theory', 'geometry', 'graph', 'greedy',
    'hash function', 'hash table', 'heap (priority queue)', 'interactive', 'line sweep', 'linked list', 'math',
    'matrix', 'memoization', 'merge sort', 'minimum spanning tree', 'monotonic queue', 'monotonic stack',
    'number theory', 'ordered set', 'prefix sum', 'probability and statistics', 'queue', 'quickselect',
    'radix sort', 'randomized', 'recursion', 'rolling hash', 'segment tree', 'shortest path', 'simulation',
    'sliding window', 'sorting', 'stack', 'string', 'string matching', 'strongly connected component',
    'suffix array', 'topological sort', 'tree', 'trie', 'two pointers', 'union find'
]

def clean_json_text(text: str) -> str:
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def extract_intent(message: str) -> dict:
    prompt = f"""
You are an AI intent extraction system.

Convert the user request into STRICT JSON with:
- topics (dictionary where keys are topic names and values are objects with "count" and "difficulty")
- topics structure: {{ "topic_name": {{ "count": int/null, "difficulty": "easy"/"medium"/"hard"/"any" }} }}
- total_questions (int, if specified)
- duration_days (int, default 21 if missing)

User request:
"{message}"

Rules:
- If counts conflict, preserve explicit counts
- If duration missing → assume 21 days
- Output ONLY the JSON object, no markdown formatting.
- Map user terms to the closest VALID TAG from the list below.
- If the user asks for generic questions (e.g. "dsa", "leetcode", "questions") WITHOUT specific topics, use "dsa" as the topic key.
- If specific valid tags are identified (e.g. "math"), do NOT output generic "dsa" unless explicitly requested as a separate category.
- Valid Tags: {", ".join(VALID_TAGS)}

Example 1: "I want mathematical questions" -> topics: {{ "math": ... }}
Example 2: "120 dsa questions" -> topics: {{ "dsa": ... }}
Example 3: "10 dsa questions, specifically math" -> topics: {{ "math": ... }}
Example 4: "10 dsa questions and 5 math questions" -> topics: {{ "dsa": {{ "count": 10... }}, "math": {{ "count": 5... }} }}

Example Output:
{{
  "topics": {{
    "two pointers": {{ "count": 2, "difficulty": "easy" }},
    "linked list": {{ "count": 48, "difficulty": "hard" }}
  }},
  "total_questions": 50,
  "duration_days": 21
}}
"""

    res = model.generate_content(prompt)
    if not res.text:
         raise ValueError("Empty response from Gemini API")
    
    cleaned_text = clean_json_text(res.text)
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        # Fallback: try to find the first '{' and last '}'
        start = cleaned_text.find('{')
        end = cleaned_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned_text[start:end+1])
            except:
                pass
        print(f"Failed to parse JSON: {cleaned_text}")
        raise ValueError(f"Invalid JSON response from model: {e}")
