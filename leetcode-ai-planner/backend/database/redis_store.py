"""
Redis memory store for conversation history
"""
import json
import redis
from typing import List, Dict
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD


class RedisChatMemoryStore:
    """Redis-backed chat memory for conversation history"""
    
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                username=REDIS_USERNAME,
                password=REDIS_PASSWORD,
            )
            self.redis.ping()
            print("✅ Redis Cloud connected successfully")
        except Exception as e:
            print(f"❌ Redis connection FAILED: {e}")
            self.redis = None
    
    def _key(self, session_id: str) -> str:
        return f"bhindi:chat:session:{session_id}"
    
    def get(self, session_id: str) -> List[Dict[str, str]]:
        if not self.redis:
            return []
        
        try:
            data = self.redis.get(self._key(session_id))
            history = json.loads(data) if data else []
            print(f"📥 Loaded {len(history)} messages from Redis for session '{session_id}'")
            return history
        except Exception as e:
            print(f"❌ Error loading from Redis: {e}")
            return []
    
    def set(self, session_id: str, history: List[Dict[str, str]]) -> None:
        if not self.redis:
            return
        
        try:
            self.redis.set(
                self._key(session_id),
                json.dumps(history),
                ex=60 * 60 * 24  # 24 hours TTL
            )
            print(f"💾 Saved {len(history)} messages to Redis")
        except Exception as e:
            print(f"❌ Error saving to Redis: {e}")
    
    def clear(self, session_id: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.delete(self._key(session_id))
            print(f"🗑️  Cleared history for session '{session_id}'")
        except Exception as e:
            print(f"❌ Error clearing Redis: {e}")


# Singleton instance
memory_store = RedisChatMemoryStore()
