"""
RTC Scholar - Ultra-Resilient Production RAG with 6 Provider Redundancy
========================================================================
2x Groq + 3x OpenRouter (different free models) + Exponential Backoff
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import re
from typing import List, Optional, Tuple
from collections import Counter
import time
import math
from threading import Lock
import logging

from knowledge_base import KNOWLEDGE_BASE

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# ULTRA-RESILIENT MULTI-PROVIDER LLM MANAGER
# ============================================
class UltraResilientLLM:
    """6 providers with smart fallback, rate limit tracking, and exponential backoff"""
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.current_provider_index = 0
        self.lock = Lock()
        self.rate_limit_tracker = {}  # Track when providers are rate-limited
        self.provider_failures = {}  # Track consecutive failures
        
        logger.info(f"✓ Initialized {len(self.providers)} LLM providers")
        for p in self.providers:
            logger.info(f"  - {p['name']}: {'✅ Configured' if p['key'] else '❌ No key'}")
            self.rate_limit_tracker[p['name']] = 0
            self.provider_failures[p['name']] = 0
    
    def _initialize_providers(self) -> List[dict]:
        """Initialize all available providers with diverse free models"""
        providers = []
        
        # ==================== GROQ PROVIDERS ====================
        # GROQ 1 - Primary (llama-3.1-8b-instant - fastest)
        groq_key_1 = os.environ.get('GROQ_API_KEY_1', '')
        if groq_key_1:
            providers.append({
                'name': 'Groq-1-Fast',
                'key': groq_key_1,
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 300,
                'temperature': 0.3,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                }
            })
        
        # GROQ 2 - Secondary (mixtral-8x7b - more capable)
        groq_key_2 = os.environ.get('GROQ_API_KEY_2', '')
        if groq_key_2:
            providers.append({
                'name': 'Groq-2-Smart',
                'key': groq_key_2,
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'mixtral-8x7b-32768',
                'max_tokens': 300,
                'temperature': 0.3,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                }
            })
        
        # ==================== OPENROUTER PROVIDERS ====================
        # OpenRouter 1 - LLaMA 3.2 3B (balanced, reliable)
        openrouter_key_1 = os.environ.get('OPENROUTER_API_KEY_1', '')
        if openrouter_key_1:
            providers.append({
                'name': 'OpenRouter-1-Llama',
                'key': openrouter_key_1,
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'model': 'meta-llama/llama-3.2-3b-instruct:free',
                'max_tokens': 250,
                'temperature': 0.3,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://rtc-scholar.onrender.com',
                    'X-Title': 'RTC-Scholar'
                }
            })
        
        # OpenRouter 2 - Mistral 7B (good for concise answers)
        openrouter_key_2 = os.environ.get('OPENROUTER_API_KEY_2', '')
        if openrouter_key_2:
            providers.append({
                'name': 'OpenRouter-2-Mistral',
                'key': openrouter_key_2,
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'model': 'mistralai/mistral-7b-instruct:free',
                'max_tokens': 250,
                'temperature': 0.3,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://rtc-scholar.onrender.com',
                    'X-Title': 'RTC-Scholar'
                }
            })
        
        # OpenRouter 3 - Phi-3 Mini (Microsoft, lightweight)
        openrouter_key_3 = os.environ.get('OPENROUTER_API_KEY_3', '')
        if openrouter_key_3:
            providers.append({
                'name': 'OpenRouter-3-Phi',
                'key': openrouter_key_3,
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'model': 'microsoft/phi-3-mini-128k-instruct:free',
                'max_tokens': 250,
                'temperature': 0.3,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://rtc-scholar.onrender.com',
                    'X-Title': 'RTC-Scholar'
                }
            })
        
        return providers
    
    def _is_rate_limited(self, provider_name: str) -> bool:
        """Check if provider is currently rate-limited"""
        rate_limit_until = self.rate_limit_tracker.get(provider_name, 0)
        return time.time() < rate_limit_until
    
    def _mark_rate_limited(self, provider_name: str, duration: int = 60):
        """Mark provider as rate-limited for duration seconds"""
        self.rate_limit_tracker[provider_name] = time.time() + duration
        logger.warning(f"⏱️  {provider_name} rate-limited for {duration}s")
    
    def _mark_failure(self, provider_name: str):
        """Track consecutive failures"""
        self.provider_failures[provider_name] = self.provider_failures.get(provider_name, 0) + 1
    
    def _mark_success(self, provider_name: str):
        """Reset failure counter on success"""
        self.provider_failures[provider_name] = 0
    
    def call(self, prompt: str, context: str) -> Tuple[str, str]:
        """Call LLM with ultra-resilient fallback strategy"""
        
        if not self.providers:
            return ("⚠️ No LLM provider configured. Please add API keys.", "error")
        
        system_prompt = f"""You are RTC Scholar, a friendly AI assistant for Rathinam Technical Campus (RTC), Coimbatore.

KNOWLEDGE BASE:
{context}

INSTRUCTIONS:
- Answer using ONLY the information from the knowledge base
- Be conversational, warm, and helpful
- Keep responses under 100 words
- If the knowledge base doesn't contain the answer, politely say so
- Don't make up information

Remember: You represent RTC, so be professional yet approachable."""

        # Try all providers with exponential backoff
        max_attempts = len(self.providers) * 2  # Try each provider twice
        
        for attempt in range(max_attempts):
            provider = self._get_next_available_provider()
            
            if not provider:
                # All providers are rate-limited or unavailable
                wait_time = min(2 ** (attempt // len(self.providers)), 8)  # Max 8s
                logger.warning(f"⏳ All providers busy, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Exponential backoff for retries
            retry_delay = min(2 ** self.provider_failures.get(provider['name'], 0), 4)
            
            try:
                logger.info(f"🔄 Attempt {attempt+1}/{max_attempts}: {provider['name']}...")
                
                response = self._call_openai_compatible(provider, system_prompt, prompt)
                
                if response:
                    self._mark_success(provider['name'])
                    logger.info(f"✅ Success with {provider['name']}")
                    return (response, "ok")
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    self._mark_rate_limited(provider['name'], duration=60)
                    logger.warning(f"🚫 {provider['name']} rate limited")
                else:
                    self._mark_failure(provider['name'])
                    logger.warning(f"❌ {provider['name']} HTTP error: {e.response.status_code}")
                
                if retry_delay > 0 and attempt < max_attempts - 1:
                    time.sleep(retry_delay)
                    
            except requests.exceptions.Timeout:
                self._mark_failure(provider['name'])
                logger.warning(f"⏱️  {provider['name']} timeout")
                
            except Exception as e:
                self._mark_failure(provider['name'])
                logger.error(f"❌ {provider['name']} failed: {str(e)[:100]}")
        
        # All providers exhausted
        return (
            "I'm experiencing high demand right now. Please try again in a moment! 🔄 "
            "(All backup systems are currently busy)",
            "error"
        )
    
    def _get_next_available_provider(self) -> Optional[dict]:
        """Get next provider that isn't rate-limited (round-robin)"""
        with self.lock:
            if not self.providers:
                return None
            
            # Try to find a non-rate-limited provider
            attempts = 0
            while attempts < len(self.providers):
                provider = self.providers[self.current_provider_index]
                self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
                
                if not self._is_rate_limited(provider['name']) and provider['key']:
                    return provider
                
                attempts += 1
            
            # All providers are rate-limited
            return None
    
    def _call_openai_compatible(self, provider: dict, system: str, user: str) -> Optional[str]:
        """Call OpenAI-compatible API (Groq, OpenRouter)"""
        
        headers = provider['headers_fn'](provider['key'])
        
        payload = {
            'model': provider['model'],
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            'temperature': provider.get('temperature', 0.3),
            'max_tokens': provider['max_tokens']
        }
        
        response = requests.post(
            provider['url'],
            headers=headers,
            json=payload,
            timeout=30  # Increased timeout
        )
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    
    def get_status(self) -> dict:
        """Get detailed provider status"""
        configured = [p['name'] for p in self.providers if p['key']]
        available = [
            p['name'] for p in self.providers 
            if p['key'] and not self._is_rate_limited(p['name'])
        ]
        
        rate_limited = [
            name for name, until in self.rate_limit_tracker.items()
            if time.time() < until
        ]
        
        return {
            'total_providers': len(self.providers),
            'configured_providers': configured,
            'available_providers': available,
            'rate_limited_providers': rate_limited,
            'current_provider': self.providers[self.current_provider_index]['name'] if self.providers else 'None',
            'health_score': f"{len(available)}/{len(configured)}"
        }

# ============================================
# VECTOR DB WITH BM25 (Unchanged - Already Good)
# ============================================
class VectorDB:
    """BM25 document retrieval"""
    
    def __init__(self):
        self.documents = []
        self.doc_frequencies = {}
        self.avg_doc_length = 0
        self.k1 = 1.5
        self.b = 0.75
    
    def add_documents(self, docs: List[str]):
        self.documents.extend(docs)
        self._build_index()
        logger.info(f"✓ Indexed {len(docs)} documents")
    
    def _build_index(self):
        for doc in self.documents:
            terms = set(self.extract_keywords(doc))
            for term in terms:
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
        
        total_length = sum(len(self.extract_keywords(doc)) for doc in self.documents)
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
    
    def normalize_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())
    
    def extract_keywords(self, text: str) -> List[str]:
        stop_words = {
            'what', 'is', 'the', 'who', 'where', 'when', 'how', 'are', 'do',
            'does', 'about', 'tell', 'me', 'can', 'you', 'a', 'an', 'and',
            'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        normalized = self.normalize_text(text)
        return [w for w in normalized.split() if w not in stop_words and len(w) > 2]
    
    def calculate_idf(self, term: str) -> float:
        doc_freq = self.doc_frequencies.get(term, 0)
        if doc_freq == 0:
            return 0
        return math.log((len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def calculate_bm25_score(self, query_terms: List[str], doc: str) -> float:
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
# INITIALIZE COMPONENTS
# ============================================
vector_db = VectorDB()
vector_db.add_documents(KNOWLEDGE_BASE)

llm_manager = UltraResilientLLM()

logger.info(f"📚 System ready with {len(KNOWLEDGE_BASE)} documents")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with provider status"""
    provider_status = llm_manager.get_status()
    
    available_count = len(provider_status['available_providers'])
    is_healthy = available_count > 0
    
    return jsonify({
        'status': 'healthy' if is_healthy else 'degraded',
        'timestamp': time.time(),
        'documents': len(vector_db.documents),
        'llm_providers': provider_status,
        'health_details': {
            'available_providers': available_count,
            'total_providers': provider_status['total_providers'],
            'rate_limited': provider_status['rate_limited_providers']
        }
    }), 200 if is_healthy else 503

@app.route('/webhook', methods=['POST'])
def webhook():
    """Main chatbot endpoint with enhanced error handling"""
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
        
        logger.info(f"❓ Query: {query_text}")
        
        # Retrieve relevant documents
        relevant_docs = vector_db.search(query_text, top_k=4)
        
        if not relevant_docs:
            context = "No specific information found in knowledge base."
        else:
            context = "\n\n".join(relevant_docs)
        
        # Call LLM with ultra-resilient fallback
        response_text, status = llm_manager.call(query_text, context)
        
        logger.info(f"✅ Response generated ({status})")
        
        return jsonify({
            "reply": response_text,
            "status": status
        })
    
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({
            "reply": "Sorry, something went wrong. Please try again! 🤖",
            "status": "error"
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed system statistics"""
    provider_status = llm_manager.get_status()
    
    return jsonify({
        'system': 'RTC Scholar v2.0 - Ultra-Resilient',
        'documents': len(vector_db.documents),
        'providers': provider_status,
        'uptime': 'Monitored by cron job',
        'timestamp': time.time()
    })

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    logger.info("\n" + "="*60)
    logger.info("🚀 RTC Scholar v2.0 - Ultra-Resilient RAG System")
    logger.info("="*60)
    logger.info(f"📊 Documents: {len(vector_db.documents)}")
    
    status = llm_manager.get_status()
    logger.info(f"🤖 LLM Providers: {status['health_score']}")
    for provider in status['configured_providers']:
        logger.info(f"   ✓ {provider}")
    logger.info(f"🌐 Port: {port}")
    logger.info("="*60 + "\n")
    
    if not status['configured_providers']:
        logger.warning("⚠️  WARNING: No LLM providers configured!")
        logger.warning("⚠️  Configure these environment variables:")
        logger.warning("⚠️  GROQ_API_KEY_1, GROQ_API_KEY_2")
        logger.warning("⚠️  OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2, OPENROUTER_API_KEY_3")
        logger.warning("⚠️  Get Groq keys: https://console.groq.com")
        logger.warning("⚠️  Get OpenRouter keys: https://openrouter.ai\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
