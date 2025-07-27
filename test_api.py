import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
print("API Key found:", bool(api_key))

print("Looking for GEMINI_API_KEY in environment...")
print(f"Available Gemini-related env vars: {[k for k in os.environ.keys() if 'GEMINI' in k.upper()]}")
if not api_key:
    print("Warning: GEMINI_API_KEY environment variable not found")
    print("Make sure your .env file contains: GEMINI_API_KEY=your_api_key_here")
else:
    print(f"Found API key: {api_key[:10]}...")

def test_gemini_api():
    # Test a simple Gemini API call
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": "Hello, this is a test."}]}]
        }
        
        print("Testing Gemini API call...")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Gemini API is working!")
            print(f"Response: {result['candidates'][0]['content']['parts'][0]['text']}")
            return True
        else:
            print(f"❌ Gemini API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()