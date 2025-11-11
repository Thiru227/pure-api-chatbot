"""
RTC Scholar - Production RAG Backend (Failproof)
=================================================
Multi-API-key rotation, circuit breaker, retry logic, health monitoring
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import re
from typing import List, Optional
from collections import Counter
import time
import math
from threading import Lock
import logging
from datetime import datetime, timedelta

from knowledge_base import KNOWLEDGE_BASE

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FLASK APP SETUP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# API KEY MANAGER WITH ROTATION
# ============================================
class APIKeyManager:
    """Manages multiple API keys with rotation and circuit breaking"""
    
    def __init__(self):
        # Load multiple API keys from environment
        self.api_keys = self._load_api_keys()
        self.current_index = 0
        self.lock = Lock()
        
        # Circuit breaker: track failures per key
        self.failures = {key: 0 for key in self.api_keys}
        self.last_failure_time = {key: None for key in self.api_keys}
        self.max_failures = 3
        self.cooldown_period = 300  # 5 minutes
        
        logger.info(f"✓ Loaded {len(self.api_keys)} API keys")
    
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables"""
        keys = []
        
        # Primary key
        primary = os.environ.get('OPENROUTER_API_KEY', '')
        if primary:
            keys.append(primary)
        
        # Fallback keys (OPENROUTER_API_KEY_2, OPENROUTER_API_KEY_3, etc.)
        i = 2
        while True:
            key = os.environ.get(f'OPENROUTER_API_KEY_{i}', '')
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        if not keys:
            logger.warning("⚠️ No API keys configured!")
        
        return keys
    
    def get_next_key(self) -> Optional[str]:
        """Get next available API key with circuit breaker logic"""
        with self.lock:
            if not self.api_keys:
                return None
            
            attempts = 0
            max_attempts = len(self.api_keys)
            
            while attempts < max_attempts:
                key = self.api_keys[self.current_index]
                
                # Check circuit breaker
                if self._is_key_available(key):
                    return key
                
                # Move to next key
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                attempts += 1
            
            # All keys are down
            logger.error("❌ All API keys are unavailable!")
            return None
    
    def _is_key_available(self, key: str) -> bool:
        """Check if key is available (not in cooldown)"""
        if self.failures[key] < self.max_failures:
            return True
        
        last_fail = self.last_failure_time[key]
        if last_fail and (datetime.now() - last_fail).seconds > self.cooldown_period:
            # Reset after cooldown
            self.failures[key] = 0
            self.last_failure_time[key] = None
            logger.info(f"✓ API key {self._mask_key(key)} recovered from cooldown")
            return True
        
        return False
    
    def report_failure(self, key: str):
        """Report a failure for an API key"""
        with self.lock:
            self.failures[key] += 1
            self.last_failure_time[key] = datetime.now()
            
            if self.failures[key] >= self.max_failures:
                logger.warning(f"⚠️ API key {self._mask_key(key)} in cooldown")
            
            # Rotate to next key
            self.current_index = (self.current_index + 1) % len(self.api_keys)
    
    def report_success(self, key: str):
        """Report a successful call"""
        with self.lock:
            if self.failures[key] > 0:
                self.failures[key] = max(0, self.failures[key] - 1)
    
    def _mask_key(self, key: str) -> str:
        """Mask API key for logging"""
        return f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    
    def get_status(self) -> dict:
        """Get status of all keys"""
        return {
            "total_keys": len(self.api_keys),
            "available_keys": sum(1 for k in self.api_keys if self._is_key_available(k)),
            "keys_in_cooldown": sum(1 for k in self.api_keys if self.failures[k] >= self.max_failures)
        }

# ============================================
# PRODUCTION VECTOR DB WITH BM25
# ============================================
class ProductionVectorDB:
    """Production-grade retrieval with BM25 ranking"""
    
    def __init__(self):
        self.documents = []
        self.doc_frequencies = {}
        self.avg_doc_length = 0
        self.k1 = 1.5
        self.b = 0.75
        logger.info("✓ ProductionVectorDB initialized")
    
    def add_documents(self, docs: List[str]):
        """Add documents and build inverted index"""
        self.documents.extend(docs)
        self._build_index()
        logger.info(f"✓ Indexed {len(docs)} documents")
    
    def _build_index(self):
        """Build inverted index"""
        for doc in self.documents:
            terms = set(self.extract_keywords(doc))
            for term in terms:
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
        
        total_length = sum(len(self.extract_keywords(doc)) for doc in self.documents)
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
    
    def normalize_text(self, text: str) -> str:
        """Normalize text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords with stop word removal"""
        stop_words = {
            'what', 'is', 'the', 'who', 'where', 'when', 'how', 'are', 'do',
            'does', 'about', 'tell', 'me', 'can', 'you', 'a', 'an', 'and',
            'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        normalized = self.normalize_text(text)
        words = [w for w in normalized.split() if w not in stop_words and len(w) > 2]
        return words
    
    def calculate_idf(self, term: str) -> float:
        """Calculate IDF"""
        doc_freq = self.doc_frequencies.get(term, 0)
        if doc_freq == 0:
            return 0
        return math.log((len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def calculate_bm25_score(self, query_terms: List[str], doc: str) -> float:
        """Calculate BM25 score"""
        doc_terms = self.extract_keywords(doc)
        doc_length = len(doc_terms)
        
        if doc_length == 0:
            return 0
        
        doc_term_freq = Counter(doc_terms)
        score = 0
        
        for term in query_terms:
            if term not in doc_term_freq:
                continue
            
            tf = doc_term_freq[term]
            idf = self.calculate_idf(term)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 4) -> List[str]:
        """Search documents"""
        if not query or not self.documents:
            return []
        
        query_terms = self.extract_keywords(query)
        
        scored_docs = []
        for doc in self.documents:
            score = self.calculate_bm25_score(query_terms, doc)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:top_k]]

# ============================================
# LLM CALLER WITH RETRY LOGIC
# ============================================
class LLMCaller:
    """Handles LLM API calls with retry and fallback logic"""
    
    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.2-3b-instruct:free"
        self.max_retries = 3
        self.timeout = 30
    
    def call(self, prompt: str, context: str) -> str:
        """Call LLM with automatic retry and key rotation"""
        
        system_prompt = f"""You are RTC Scholar, a helpful AI assistant for Rathinam Technical Campus.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer using ONLY the context provided
2. Be accurate and concise (under 100 words)
3. If context doesn't contain the answer, say so politely
4. Be friendly and professional

Remember: You represent Rathinam Technical Campus."""

        for attempt in range(self.max_retries):
            api_key = self.api_key_manager.get_next_key()
            
            if not api_key:
                logger.error("No API keys available")
                return "⚠️ Service temporarily unavailable. Please try again shortly."
            
            try:
                response = self._make_request(api_key, system_prompt, prompt)
                
                if response:
                    self.api_key_manager.report_success(api_key)
                    return response
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                self.api_key_manager.report_failure(api_key)
                
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return "Sorry, I'm having trouble responding right now. Please try again! 🔄"
    
    def _make_request(self, api_key: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Make single API request"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://render.com",
            "X-Title": "RTC-Scholar"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        
        # Handle specific error codes
        if response.status_code in [401, 402, 403]:
            logger.warning(f"API key issue: {response.status_code}")
            raise Exception(f"API key error: {response.status_code}")
        
        if response.status_code == 429:
            logger.warning("Rate limit hit")
            raise Exception("Rate limit")
        
        logger.error(f"API error {response.status_code}: {response.text[:200]}")
        raise Exception(f"API error: {response.status_code}")

# ============================================
# INITIALIZE COMPONENTS
# ============================================
vector_db = ProductionVectorDB()
vector_db.add_documents(KNOWLEDGE_BASE)

api_key_manager = APIKeyManager()
llm_caller = LLMCaller(api_key_manager)

logger.info(f"📚 System ready: {len(KNOWLEDGE_BASE)} documents")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    api_status = api_key_manager.get_status()
    
    health_data = {
        'status': 'healthy' if api_status['available_keys'] > 0 else 'degraded',
        'timestamp': time.time(),
        'documents': len(vector_db.documents),
        'api_keys': api_status,
        'uptime': time.time()
    }
    
    status_code = 200 if api_status['available_keys'] > 0 else 503
    
    return jsonify(health_data), status_code

@app.route('/webhook', methods=['POST'])
def webhook():
    """Main chatbot endpoint"""
    try:
        req = request.get_json(silent=True, force=True)
        
        if not req:
            return jsonify({
                "reply": "Invalid request format",
                "status": "error"
            }), 400
        
        # Extract query from various formats
        query_text = (
            req.get('queryResult', {}).get('queryText') if isinstance(req.get('queryResult'), dict)
            else req.get('message') or req.get('query') or req.get('text')
        )
        
        if not query_text:
            return jsonify({
                "reply": "No query text provided",
                "status": "error"
            }), 400
        
        logger.info(f"Query: {query_text}")
        
        # RAG retrieval
        relevant_docs = vector_db.search(query_text, top_k=4)
        context = "\n\n".join(relevant_docs) if relevant_docs else "No relevant information found."
        
        # LLM call with failover
        response_text = llm_caller.call(query_text, context)
        
        return jsonify({
            "reply": response_text,
            "status": "ok"
        })
    
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({
            "reply": "Sorry, something went wrong. Please try again! 🤖",
            "status": "error"
        }), 500

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    logger.info("="*50)
    logger.info("🚀 RTC Scholar Starting")
    logger.info(f"📊 Documents: {len(vector_db.documents)}")
    logger.info(f"🔑 API Keys: {len(api_key_manager.api_keys)}")
    logger.info(f"🌐 Port: {port}")
    logger.info("="*50)
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True  # Handle concurrent requests
    )
