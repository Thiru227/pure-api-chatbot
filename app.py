"""
RTC Scholar - Production RAG with Multiple Provider Support
============================================================
Supports: Groq (primary), OpenRouter (fallback), and more
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
# MULTI-PROVIDER LLM MANAGER
# ============================================
class MultiProviderLLM:
    """Supports multiple LLM providers with automatic fallback"""
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.current_provider_index = 0
        self.lock = Lock()
        
        logger.info(f"✓ Initialized {len(self.providers)} LLM providers")
        for p in self.providers:
            logger.info(f"  - {p['name']}: {'✅ Configured' if p['key'] else '❌ No key'}")
    
    def _initialize_providers(self) -> List[dict]:
        """Initialize all available providers"""
        providers = []
        
        # GROQ - Best for free tier
        groq_key = os.environ.get('GROQ_API_KEY', '')
        if groq_key:
            providers.append({
                'name': 'Groq',
                'key': groq_key,
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 300,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                }
            })
        
        # OpenRouter - Fallback
        openrouter_key = os.environ.get('OPENROUTER_API_KEY', '')
        if openrouter_key:
            providers.append({
                'name': 'OpenRouter',
                'key': openrouter_key,
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'model': 'meta-llama/llama-3.2-3b-instruct:free',
                'max_tokens': 250,
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://render.com',
                    'X-Title': 'RTC-Scholar'
                }
            })
        
        # Google Gemini - Another fallback option
        gemini_key = os.environ.get('GEMINI_API_KEY', '')
        if gemini_key:
            providers.append({
                'name': 'Gemini',
                'key': gemini_key,
                'url': f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}',
                'model': 'gemini-pro',
                'max_tokens': 300,
                'is_gemini': True,  # Special handling
                'headers_fn': lambda key: {
                    'Content-Type': 'application/json'
                }
            })
        
        return providers
    
    def call(self, prompt: str, context: str) -> Tuple[str, str]:
        """Call LLM with automatic provider fallback"""
        
        if not self.providers:
            return ("⚠️ No LLM provider configured. Please add GROQ_API_KEY or OPENROUTER_API_KEY.", "error")
        
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

        # Try each provider
        for attempt in range(len(self.providers)):
            provider = self._get_next_provider()
            
            if not provider or not provider['key']:
                continue
            
            try:
                logger.info(f"🔄 Trying {provider['name']}...")
                
                # Special handling for Gemini
                if provider.get('is_gemini'):
                    response = self._call_gemini(provider, system_prompt, prompt)
                else:
                    response = self._call_openai_compatible(provider, system_prompt, prompt)
                
                if response:
                    logger.info(f"✅ Success with {provider['name']}")
                    return (response, "ok")
                    
            except Exception as e:
                logger.warning(f"❌ {provider['name']} failed: {str(e)[:100]}")
                continue
        
        # All providers failed
        return ("Sorry, I'm having trouble connecting right now. Please try again in a moment! 🔄", "error")
    
    def _get_next_provider(self) -> Optional[dict]:
        """Get next provider (round-robin)"""
        with self.lock:
            if not self.providers:
                return None
            
            provider = self.providers[self.current_provider_index]
            self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
            return provider
    
    def _call_openai_compatible(self, provider: dict, system: str, user: str) -> Optional[str]:
        """Call OpenAI-compatible API (Groq, OpenRouter)"""
        
        headers = provider['headers_fn'](provider['key'])
        
        payload = {
            'model': provider['model'],
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            'temperature': 0.3,
            'max_tokens': provider['max_tokens']
        }
        
        response = requests.post(
            provider['url'],
            headers=headers,
            json=payload,
            timeout=25
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        
        # Log error but don't crash
        logger.error(f"{provider['name']} error {response.status_code}: {response.text[:200]}")
        return None
    
    def _call_gemini(self, provider: dict, system: str, user: str) -> Optional[str]:
        """Call Google Gemini API"""
        
        headers = provider['headers_fn'](provider['key'])
        
        # Gemini has different payload format
        payload = {
            'contents': [{
                'parts': [{
                    'text': f"{system}\n\nUser question: {user}"
                }]
            }],
            'generationConfig': {
                'temperature': 0.3,
                'maxOutputTokens': provider['max_tokens']
            }
        }
        
        response = requests.post(
            provider['url'],
            headers=headers,
            json=payload,
            timeout=25
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        return None
    
    def get_status(self) -> dict:
        """Get provider status"""
        return {
            'total_providers': len(self.providers),
            'configured_providers': [p['name'] for p in self.providers if p['key']],
            'current_provider': self.providers[self.current_provider_index]['name'] if self.providers else 'None'
        }

# ============================================
# VECTOR DB WITH BM25
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

llm_manager = MultiProviderLLM()

logger.info(f"📚 System ready with {len(KNOWLEDGE_BASE)} documents")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    provider_status = llm_manager.get_status()
    
    is_healthy = len(provider_status['configured_providers']) > 0
    
    return jsonify({
        'status': 'healthy' if is_healthy else 'degraded',
        'timestamp': time.time(),
        'documents': len(vector_db.documents),
        'llm_providers': provider_status
    }), 200 if is_healthy else 503

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
        
        logger.info(f"❓ Query: {query_text}")
        
        # Retrieve relevant documents
        relevant_docs = vector_db.search(query_text, top_k=4)
        
        if not relevant_docs:
            context = "No specific information found in knowledge base."
        else:
            context = "\n\n".join(relevant_docs)
        
        # Call LLM with fallback
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

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    logger.info("\n" + "="*60)
    logger.info("🚀 RTC Scholar - Production RAG System")
    logger.info("="*60)
    logger.info(f"📊 Documents: {len(vector_db.documents)}")
    
    status = llm_manager.get_status()
    logger.info(f"🤖 LLM Providers: {', '.join(status['configured_providers']) if status['configured_providers'] else 'None configured!'}")
    logger.info(f"🌐 Port: {port}")
    logger.info("="*60 + "\n")
    
    if not status['configured_providers']:
        logger.warning("⚠️  WARNING: No LLM providers configured!")
        logger.warning("⚠️  Add GROQ_API_KEY environment variable")
        logger.warning("⚠️  Get free key: https://console.groq.com\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
