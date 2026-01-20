"""
Configuration and environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis-13657.c80.us-east-1-2.ec2.cloud.redislabs.com")
REDIS_PORT = int(os.getenv("REDIS_PORT", "13657"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "qDFYNsEVRSi6t2Z8CwhUdiG2JJDtTH3V")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

LEETCODE_DATASET_PATH = os.path.join(DATA_DIR, "leetcode_dataset.csv")
COMPANY_DATASET_PATH = os.path.join(DATA_DIR, "COMBINED_DSA_All_Companies.xlsx")

LEETCODE_VECTOR_INDEX = os.path.join(DATA_DIR, "leetcode_faiss_index.bin")
LEETCODE_METADATA = os.path.join(DATA_DIR, "leetcode_metadata.pkl")
COMPANY_VECTOR_INDEX = os.path.join(DATA_DIR, "company_faiss_index.bin")
COMPANY_METADATA = os.path.join(DATA_DIR, "company_metadata.pkl")

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
