from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import time

app = Flask(__name__)
CORS(app)

# Hugging Face Model URL for NLLB
MODEL_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"

# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

if not HF_TOKEN:
    print("WARNING: HUGGINGFACE_TOKEN environment variable not set!")
    print("Please set your Hugging Face API token in Render's environment variables")

def translate_to_telugu(text):
    """
    Translates English text to Telugu using NLLB model via Hugging Face API
    """
    try:
        if not HF_TOKEN:
            return "Error: Hugging Face token not configured. Please contact administrator."
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # NLLB model payload for English to Telugu translation
        payload = {
            "inputs": text,
            "parameters": {
                "src_lang": "eng_Latn",  # English (Latin script)
                "tgt_lang": "tel_Telu"   # Telugu (Telugu script)
            },
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
        
        print(f"Translating: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(MODEL_URL, json=payload, headers=headers, timeout=60)
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    return "Translation request timed out. Please try with shorter text."
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result}")
            
            # Handle different response formats from Hugging Face
            if isinstance(result, list) and len(result) > 0:
                if 'translation_text' in result[0]:
                    translation = result[0]['translation_text']
                    print(f"Translation successful: {translation}")
                    return translation
                elif 'generated_text' in result[0]:
                    translation = result[0]['generated_text']
                    print(f"Translation successful: {translation}")
                    return translation
                else:
                    print(f"Unexpected response format: {result}")
                    return "Translation completed but format unexpected"
            
            elif isinstance(result, dict):
                if 'translation_text' in result:
                    translation = result['translation_text']
                    print(f"Translation successful: {translation}")
                    return translation
                elif 'generated_text' in result:
                    translation = result['generated_text']
                    print(f"Translation successful: {translation}")
                    return translation
                elif 'error' in result:
                    print(f"API returned error: {result['error']}")
                    return f"Translation error: {result['error']}"
            
            return str(result)
        
        elif response.status_code == 503:
            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"Model loading (503): {error_detail}")
            return "Model is loading, please wait a moment and try again"
        
        elif response.status_code == 401:
            print("Authentication failed - check Hugging Face token")
            return "Authentication failed. Please check API configuration."
        
        elif response.status_code == 429:
            print("Rate limit exceeded")
            return "Too many requests. Please wait a moment and try again."
        
        else:
            error_detail = response.text
            print(f"API Error {response.status_code}: {error_detail}")
            return f"Translation service temporarily unavailable (Error {response.status_code})"
    
    except requests.exceptions.Timeout:
        print("Request timed out")
        return "Translation request timed out. Please try again with shorter text."
    
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return "Network connection error. Please try again."
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return "Network error occurred. Please try again."
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again."

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with service information"""
    return jsonify({
        "message": "NLLB Language Translator Service is running!",
        "model": "facebook/nllb-200-distilled-600M",
        "languages": "English to Telugu",
        "version": "1.0.0",
        "endpoints": {
            "/": "Service information",
            "/health": "Health check",
            "/process": "POST - Translate text",
            "/test": "POST - Test translation"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "nllb-translator",
        "model": "facebook/nllb-200-distilled-600M",
        "token_configured": bool(HF_TOKEN),
        "timestamp": time.time()
    })

@app.route('/process', methods=['POST'])
def process_text():
    """Main translation endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "expected_format": {"sentence": "text to translate"}
            }), 400
        
        if 'sentence' not in data:
            return jsonify({
                "error": "Missing 'sentence' field in request",
                "expected_format": {"sentence": "text to translate"}
            }), 400
        
        sentence = data['sentence']
        
        if not sentence or not sentence.strip():
            return jsonify({
                "error": "Empty sentence provided"
            }), 400
        
        sentence = sentence.strip()
        
        # Check text length (NLLB has limits)
        if len(sentence) > 1000:
            return jsonify({
                "error": "Text too long. Please limit to 1000 characters."
            }), 400
        
        print(f"Processing translation request for: '{sentence[:100]}{'...' if len(sentence) > 100 else ''}'")
        
        # Translate the sentence
        translated = translate_to_telugu(sentence)
        
        response_data = {
            "original_sentence": sentence,
            "processed_sentence": translated,
            "status": "success",
            "model": "facebook/nllb-200-distilled-600M",
            "language_pair": "English → Telugu"
        }
        
        print(f"Translation completed successfully")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in process_text: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": "Please try again later"
        }), 500

@app.route('/test', methods=['POST', 'GET'])
def test_translation():
    """Test endpoint for debugging"""
    try:
        if request.method == 'GET':
            test_text = "Hello, how are you today?"
        else:
            data = request.get_json()
            test_text = data.get('sentence', 'Hello, how are you today?')
        
        print(f"Testing translation with: '{test_text}'")
        result = translate_to_telugu(test_text)
        
        return jsonify({
            "test_input": test_text,
            "test_output": result,
            "token_configured": bool(HF_TOKEN),
            "model": "facebook/nllb-200-distilled-600M",
            "status": "test_completed"
        })
    except Exception as e:
        print(f"Error in test endpoint: {e}")
        return jsonify({
            "error": str(e),
            "token_configured": bool(HF_TOKEN)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/process", "/test"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("🚀 NLLB Translation Service Starting")
    print("=" * 50)
    print(f"🌐 Port: {port}")
    print(f"🔑 Token configured: {bool(HF_TOKEN)}")
    print(f"🤖 Model: facebook/nllb-200-distilled-600M")
    print(f"🌍 Translation: English → Telugu")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=False)