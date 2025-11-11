"""
RTC Scholar - DIAGNOSTIC VERSION
=================================
This version includes detailed logging to diagnose the rate limit issue
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
import logging
import json

from knowledge_base import KNOWLEDGE_BASE

# ============================================
# ENHANCED LOGGING
# ============================================
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# API KEY TESTER
# ============================================
class APIKeyTester:
    """Test API keys and diagnose issues"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        logger.info(f"🔑 Loaded {len(self.api_keys)} API keys")
        
    def _load_api_keys(self) -> List[str]:
        """Load API keys"""
        keys = []
        primary = os.environ.get('OPENROUTER_API_KEY', '')
        if primary:
            keys.append(('PRIMARY', primary))
        
        i = 2
        while True:
            key = os.environ.get(f'OPENROUTER_API_KEY_{i}', '')
            if key:
                keys.append((f'KEY_{i}', key))
                i += 1
            else:
                break
        
        return keys
    
    def test_key(self, name: str, api_key: str) -> dict:
        """Test a single API key with detailed diagnostics"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing {name}")
        logger.info(f"🔑 Key: {api_key[:15]}...{api_key[-8:]}")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://render.com",
            "X-Title": "RTC-Scholar"
        }
        
        # Minimal test payload
        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 10
        }
        
        result = {
            'name': name,
            'key_preview': f"{api_key[:15]}...{api_key[-8:]}",
            'status': 'unknown',
            'status_code': None,
            'error': None,
            'response_time': None,
            'raw_response': None
        }
        
        try:
            logger.info("📤 Sending request...")
            start_time = time.time()
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            result['response_time'] = time.time() - start_time
            result['status_code'] = response.status_code
            
            logger.info(f"📥 Status Code: {response.status_code}")
            logger.info(f"⏱️  Response Time: {result['response_time']:.2f}s")
            logger.info(f"📋 Headers: {dict(response.headers)}")
            
            # Log raw response
            try:
                result['raw_response'] = response.json()
                logger.info(f"📄 Response Body: {json.dumps(result['raw_response'], indent=2)}")
            except:
                result['raw_response'] = response.text[:500]
                logger.info(f"📄 Response Text: {response.text[:500]}")
            
            if response.status_code == 200:
                result['status'] = 'SUCCESS ✅'
                logger.info("✅ KEY WORKS!")
            
            elif response.status_code == 401:
                result['status'] = 'INVALID KEY ❌'
                result['error'] = 'Authentication failed - key is invalid'
                logger.error("❌ Invalid API Key")
            
            elif response.status_code == 402:
                result['status'] = 'NO CREDITS ⚠️'
                result['error'] = 'Insufficient credits'
                logger.error("⚠️ No credits available")
            
            elif response.status_code == 429:
                result['status'] = 'RATE LIMITED ⏸️'
                result['error'] = 'Rate limit hit'
                
                # Parse rate limit details
                rate_limit_info = {
                    'limit': response.headers.get('x-ratelimit-limit'),
                    'remaining': response.headers.get('x-ratelimit-remaining'),
                    'reset': response.headers.get('x-ratelimit-reset')
                }
                result['rate_limit_info'] = rate_limit_info
                logger.warning(f"⏸️ Rate Limit: {rate_limit_info}")
            
            else:
                result['status'] = f'ERROR {response.status_code} ❌'
                result['error'] = response.text[:200]
                logger.error(f"❌ Unexpected status: {response.status_code}")
        
        except requests.exceptions.Timeout:
            result['status'] = 'TIMEOUT ⏱️'
            result['error'] = 'Request timed out after 30s'
            logger.error("⏱️ Request timeout")
        
        except Exception as e:
            result['status'] = 'EXCEPTION 💥'
            result['error'] = str(e)
            logger.error(f"💥 Exception: {e}")
        
        logger.info(f"{'='*60}\n")
        return result
    
    def test_all_keys(self) -> dict:
        """Test all API keys"""
        logger.info("\n" + "="*70)
        logger.info("🧪 STARTING API KEY DIAGNOSTICS")
        logger.info("="*70 + "\n")
        
        results = []
        
        for name, key in self.api_keys:
            result = self.test_key(name, key)
            results.append(result)
            time.sleep(2)  # Wait between tests
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 DIAGNOSTIC SUMMARY")
        logger.info("="*70)
        
        working = sum(1 for r in results if '✅' in r['status'])
        rate_limited = sum(1 for r in results if '⏸️' in r['status'])
        invalid = sum(1 for r in results if '❌' in r['status'])
        
        logger.info(f"✅ Working Keys: {working}/{len(results)}")
        logger.info(f"⏸️ Rate Limited: {rate_limited}/{len(results)}")
        logger.info(f"❌ Invalid/Error: {invalid}/{len(results)}")
        logger.info("="*70 + "\n")
        
        return {
            'total_keys': len(results),
            'working': working,
            'rate_limited': rate_limited,
            'invalid': invalid,
            'details': results
        }

# ============================================
# SIMPLE VECTOR DB
# ============================================
class SimpleVectorDB:
    """Simplified for testing"""
    
    def __init__(self):
        self.documents = []
    
    def add_documents(self, docs: List[str]):
        self.documents.extend(docs)
        logger.info(f"✓ Loaded {len(docs)} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Simple keyword search"""
        query_lower = query.lower()
        
        scored = []
        for doc in self.documents:
            score = sum(1 for word in query_lower.split() if word in doc.lower())
            if score > 0:
                scored.append((score, doc))
        
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k]]

# ============================================
# INITIALIZE
# ============================================
vector_db = SimpleVectorDB()
vector_db.add_documents(KNOWLEDGE_BASE)

api_tester = APIKeyTester()

# ============================================
# ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'documents': len(vector_db.documents),
        'api_keys_configured': len(api_tester.api_keys)
    }), 200

@app.route('/test-keys', methods=['GET'])
def test_keys():
    """Test all API keys and return diagnostic report"""
    logger.info("🚀 Starting API key diagnostics...")
    
    results = api_tester.test_all_keys()
    
    return jsonify(results), 200

@app.route('/test-single', methods=['POST'])
def test_single():
    """Test a single API call with detailed logging"""
    try:
        req = request.get_json()
        query = req.get('query', 'Who is the principal?')
        
        if not api_tester.api_keys:
            return jsonify({
                'error': 'No API keys configured',
                'status': 'error'
            }), 400
        
        # Use first available key
        name, api_key = api_tester.api_keys[0]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing single request with {name}")
        logger.info(f"❓ Query: {query}")
        
        # Get context
        docs = vector_db.search(query, top_k=2)
        context = "\n".join(docs) if docs else "No context"
        
        logger.info(f"📚 Context length: {len(context)} chars")
        
        # Make request
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://render.com",
            "X-Title": "RTC-Scholar"
        }
        
        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Context: {context}"
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }
        
        logger.info("📤 Sending request to OpenRouter...")
        start_time = time.time()
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        elapsed = time.time() - start_time
        
        logger.info(f"📥 Response received in {elapsed:.2f}s")
        logger.info(f"📊 Status Code: {response.status_code}")
        logger.info(f"📋 Response Headers: {dict(response.headers)}")
        
        result = {
            'query': query,
            'status_code': response.status_code,
            'response_time': elapsed,
            'headers': dict(response.headers),
        }
        
        if response.status_code == 200:
            data = response.json()
            result['success'] = True
            result['response'] = data['choices'][0]['message']['content']
            logger.info(f"✅ SUCCESS: {result['response']}")
        else:
            result['success'] = False
            result['error'] = response.text
            logger.error(f"❌ FAILED: {response.text}")
        
        logger.info(f"{'='*60}\n")
        
        return jsonify(result), response.status_code
    
    except Exception as e:
        logger.error(f"💥 Exception: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Regular webhook - logs everything"""
    try:
        req = request.get_json(silent=True, force=True)
        
        logger.info(f"\n{'='*60}")
        logger.info("📥 WEBHOOK REQUEST")
        logger.info(f"📋 Payload: {json.dumps(req, indent=2)}")
        
        query_text = (
            req.get('queryResult', {}).get('queryText') if isinstance(req.get('queryResult'), dict)
            else req.get('message') or req.get('query') or req.get('text')
        )
        
        if not query_text:
            return jsonify({"reply": "No query", "status": "error"}), 400
        
        logger.info(f"❓ Query: {query_text}")
        
        # Simple response for now
        docs = vector_db.search(query_text, top_k=3)
        
        if docs:
            response = f"Found relevant info: {docs[0][:100]}..."
        else:
            response = "No relevant information found in knowledge base."
        
        logger.info(f"💬 Response: {response}")
        logger.info(f"{'='*60}\n")
        
        return jsonify({
            "reply": response,
            "status": "ok"
        })
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({"reply": "Error occurred", "status": "error"}), 500

# ============================================
# STARTUP DIAGNOSTICS
# ============================================
@app.before_request
def log_request():
    """Log every incoming request"""
    logger.debug(f"📨 {request.method} {request.path} from {request.remote_addr}")

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    logger.info("\n" + "="*70)
    logger.info("🚀 RTC Scholar - DIAGNOSTIC MODE")
    logger.info("="*70)
    logger.info(f"📊 Documents: {len(vector_db.documents)}")
    logger.info(f"🔑 API Keys: {len(api_tester.api_keys)}")
    logger.info(f"🌐 Port: {port}")
    logger.info("")
    logger.info("🧪 DIAGNOSTIC ENDPOINTS:")
    logger.info("   GET  /health        - Basic health check")
    logger.info("   GET  /test-keys     - Test all API keys")
    logger.info("   POST /test-single   - Test single request with logging")
    logger.info("   POST /webhook       - Regular webhook (with logs)")
    logger.info("="*70 + "\n")
    
    # Auto-test keys on startup
    logger.info("🔬 Running startup diagnostics...\n")
    startup_results = api_tester.test_all_keys()
    
    if startup_results['working'] > 0:
        logger.info("✅ System ready - at least one key works!")
    else:
        logger.error("❌ WARNING: No working API keys detected!")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
