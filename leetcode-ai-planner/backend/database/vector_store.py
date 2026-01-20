"""
FAISS vector database management
"""
import os
import re
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config.settings import (
    LEETCODE_DATASET_PATH, COMPANY_DATASET_PATH,
    LEETCODE_VECTOR_INDEX, LEETCODE_METADATA,
    COMPANY_VECTOR_INDEX, COMPANY_METADATA,
    EMBEDDING_MODEL_NAME
)


class VectorStore:
    """Manages vector databases for LeetCode and Company questions"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.leetcode_index, self.leetcode_df = self._load_or_build_leetcode()
        self.company_index, self.company_df = self._load_or_build_company()
    
    def _load_or_build_leetcode(self):
        """Load or build LeetCode vector database"""
        if os.path.exists(LEETCODE_VECTOR_INDEX) and os.path.exists(LEETCODE_METADATA):
            print("📂 Loading LeetCode vector database...")
            index = faiss.read_index(LEETCODE_VECTOR_INDEX)
            with open(LEETCODE_METADATA, 'rb') as f:
                metadata = pickle.load(f)
            df = pd.DataFrame(metadata["dataframe"])
            print(f"✅ LeetCode: {len(df)} questions loaded")
            return index, df
        else:
            return self._build_leetcode_db()
    
    def _build_leetcode_db(self):
        """Build LeetCode vector database from scratch"""
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
        
        embeddings = self.embedding_model.encode(semantic_texts, show_progress_bar=True, batch_size=32)
        
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
        return index, df
    
    def _load_or_build_company(self):
        """Load or build Company vector database"""
        if os.path.exists(COMPANY_VECTOR_INDEX) and os.path.exists(COMPANY_METADATA):
            print("📂 Loading Company vector database...")
            index = faiss.read_index(COMPANY_VECTOR_INDEX)
            with open(COMPANY_METADATA, 'rb') as f:
                metadata = pickle.load(f)
            df = pd.DataFrame(metadata["dataframe"])
            print(f"✅ Company: {len(df)} questions loaded")
            return index, df
        else:
            return self._build_company_db()
    
    def _build_company_db(self):
        """Build Company vector database from scratch"""
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
        
        embeddings = self.embedding_model.encode(semantic_texts, show_progress_bar=True, batch_size=32)
        
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
        return index, df


# Singleton instance
vector_store = VectorStore()
