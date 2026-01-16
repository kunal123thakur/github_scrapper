# import os
# import sys
# import json
# import re
# from datetime import date, timedelta
# from typing import List, Dict, Optional

# import pandas as pd
# import google.generativeai as genai
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import uvicorn

# # ==========================================
# # CONFIGURATION
# # ==========================================
# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL = 

# # Adjust path for standalone execution in root
# # Assuming standleetcode.py is in project root
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # Path to the dataset inside the original structure
# DATASET_PATH = os.path.join(BASE_DIR, "leetcode-ai-planner", "backend", "data", "leetcode_dataset.csv")

# # ==========================================
# # SCHEMAS
# # ==========================================
# class ChatRequest(BaseModel):
#     message: str

# # ==========================================
# # DATASET LOADER
# # ==========================================
# def load_dataset():
#     if not os.path.exists(DATASET_PATH):
#         raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
#     df = pd.read_csv(DATASET_PATH)

#     # Normalize
#     # The tags are stored as string representation of numpy array: "['Tag1' 'Tag2']"
#     # We use regex to extract content within single quotes
#     df["tags"] = df["tags"].apply(
#         lambda x: [t.lower() for t in re.findall(r"'([^']*)'", str(x))]
#     )
#     df["difficulty"] = df["difficulty"].str.lower()
#     df["problem_description"] = df["problem_description"].fillna("")

#     return df

# # ==========================================
# # VALIDATOR
# # ==========================================
# def validate_intent(intent):
#     if not intent.get("topics"):
#         raise ValueError("No valid topics found in request")

#     for topic, cfg in intent["topics"].items():
#         if cfg["count"] is not None and cfg["count"] <= 0:
#             raise ValueError(f"Invalid count for {topic}")

# # ==========================================
# # INTENT AGENT
# # ==========================================
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel(GEMINI_MODEL)

# VALID_TAGS = [
#     'array', 'backtracking', 'biconnected component', 'binary indexed tree', 'binary search', 'binary search tree',
#     'binary tree', 'bit manipulation', 'bitmask', 'brainteaser', 'breadth-first search', 'bucket sort',
#     'combinatorics', 'concurrency', 'counting', 'counting sort', 'depth-first search', 'divide and conquer',
#     'dynamic programming', 'enumeration', 'eulerian circuit', 'game theory', 'geometry', 'graph', 'greedy',
#     'hash function', 'hash table', 'heap (priority queue)', 'interactive', 'line sweep', 'linked list', 'math',
#     'matrix', 'memoization', 'merge sort', 'minimum spanning tree', 'monotonic queue', 'monotonic stack',
#     'number theory', 'ordered set', 'prefix sum', 'probability and statistics', 'queue', 'quickselect',
#     'radix sort', 'randomized', 'recursion', 'rolling hash', 'segment tree', 'shortest path', 'simulation',
#     'sliding window', 'sorting', 'stack', 'string', 'string matching', 'strongly connected component',
#     'suffix array', 'topological sort', 'tree', 'trie', 'two pointers', 'union find'
# ]

# def clean_json_text(text: str) -> str:
#     # Remove markdown code blocks
#     text = re.sub(r'```json\s*', '', text)
#     text = re.sub(r'```\s*', '', text)
#     return text.strip()

# def extract_intent(message: str) -> dict:
#     prompt = f"""
# You are an AI intent extraction system.

# Convert the user request into STRICT JSON with:
# - topics (dictionary where keys are topic names and values are objects with "count" and "difficulty")
# - topics structure: {{ "topic_name": {{ "count": int/null, "difficulty": "easy"/"medium"/"hard"/"any" }} }}
# - total_questions (int, if specified)
# - duration_days (int, default 21 if missing)

# User request:
# "{message}"

# Rules:
# - If counts conflict, preserve explicit counts
# - If duration missing → assume 21 days
# - Output ONLY the JSON object, no markdown formatting.
# - Map user terms to the closest VALID TAG from the list below.
# - If the user asks for generic questions (e.g. "dsa", "leetcode", "questions") WITHOUT specific topics, use "dsa" as the topic key.
# - If specific valid tags are identified (e.g. "math"), do NOT output generic "dsa" unless explicitly requested as a separate category.
# - Valid Tags: {", ".join(VALID_TAGS)}

# Example 1: "I want mathematical questions" -> topics: {{ "math": ... }}
# Example 2: "120 dsa questions" -> topics: {{ "dsa": ... }}
# Example 3: "10 dsa questions, specifically math" -> topics: {{ "math": ... }}
# Example 4: "10 dsa questions and 5 math questions" -> topics: {{ "dsa": {{ "count": 10... }}, "math": {{ "count": 5... }} }}

# Example Output:
# {{
#   "topics": {{
#     "two pointers": {{ "count": 2, "difficulty": "easy" }},
#     "linked list": {{ "count": 48, "difficulty": "hard" }}
#   }},
#   "total_questions": 50,
#   "duration_days": 21
# }}
# """

#     res = model.generate_content(prompt)
#     if not res.text:
#          raise ValueError("Empty response from Gemini API")
    
#     cleaned_text = clean_json_text(res.text)
    
#     try:
#         return json.loads(cleaned_text)
#     except json.JSONDecodeError as e:
#         # Fallback: try to find the first '{' and last '}'
#         start = cleaned_text.find('{')
#         end = cleaned_text.rfind('}')
#         if start != -1 and end != -1:
#             try:
#                 return json.loads(cleaned_text[start:end+1])
#             except:
#                 pass
#         print(f"Failed to parse JSON: {cleaned_text}")
#         raise ValueError(f"Invalid JSON response from model: {e}")

# # ==========================================
# # PLANNER AGENT
# # ==========================================
# def plan_distribution(intent: dict) -> dict:
#     topics = intent["topics"]

#     explicit_total = sum(
#         t["count"] for t in topics.values() if t["count"] is not None
#     )

#     remaining_topics = [
#         k for k, v in topics.items() if v["count"] is None
#     ]

#     if intent.get("total_questions"):
#         remaining = intent["total_questions"] - explicit_total

#         if remaining < 0:
#             raise ValueError("Topic-wise counts exceed total questions")

#         if remaining_topics:
#             per_topic = remaining // len(remaining_topics)
#             for t in remaining_topics:
#                 topics[t]["count"] = per_topic

#     return {
#         "topics": topics,
#         "duration_days": intent.get("duration_days", 21)
#     }

# # ==========================================
# # FILTER AGENT
# # ==========================================
# def filter_questions(df, plan):
#     result = []

#     for topic, cfg in plan["topics"].items():
#         if topic.lower() in ["any", "dsa", "all", "questions", "leetcode"]:
#             temp = df
#         else:
#             temp = df[df["tags"].apply(lambda t: topic.lower() in t)]

#         if cfg["difficulty"] != "any":
#             temp = temp[temp["difficulty"] == cfg["difficulty"]]

#         if len(temp) < cfg["count"]:
#             raise ValueError(
#                 f"Only {len(temp)} questions available for {topic}, "
#                 f"but {cfg['count']} requested."
#             )

#         result.append(temp.sample(cfg["count"]))

#     return result

# # ==========================================
# # SCHEDULER AGENT
# # ==========================================
# def schedule(dfs, days):
#     questions = []
#     for df in dfs:
#         questions.extend(df.to_dict(orient="records"))

#     start = date.today()
#     schedule_map = {}

#     for i, q in enumerate(questions):
#         day = start + timedelta(days=i % days)
#         schedule_map.setdefault(str(day), []).append({
#             "title": q["task_id"].replace("-", " ").title(),
#             "difficulty": q["difficulty"].title(),
#             "topic": q["tags"][0].title(),
#             "url": f"https://leetcode.com/problems/{q['task_id']}/"
#         })

#     return schedule_map

# # ==========================================
# # RESPONSE AGENT
# # ==========================================
# def build_response(schedule_map):
#     return {
#         "total_days": len(schedule_map),
#         "schedule": [
#             {"date": d, "questions": qs}
#             for d, qs in sorted(schedule_map.items())
#         ]
#     }

# # ==========================================
# # MAIN APP
# # ==========================================
# app = FastAPI()

# # Load dataset once at startup
# try:
#     df = load_dataset()
#     print(f"Dataset loaded successfully with {len(df)} questions.")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     df = None

# @app.post("/chat")
# def chat(req: ChatRequest):
#     if df is None:
#         raise HTTPException(status_code=500, detail="Dataset not loaded")
        
#     try:
#         intent = extract_intent(req.message)
#         validate_intent(intent)

#         plan = plan_distribution(intent)
#         filtered = filter_questions(df, plan)
#         scheduled = schedule(filtered, plan["duration_days"])

#         return build_response(scheduled)

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
