"""
Quick test script for your deployed Railway API.
Replace YOUR_RAILWAY_URL with your actual Railway URL.
"""

import requests
import json

# Replace this with your actual Railway URL
RAILWAY_URL = "https://your-app.railway.app"

def test_deployed_api():
    print("🧪 Testing Railway Deployed API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{RAILWAY_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health Check: PASSED")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health Check: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Health Check: FAILED (Error: {e})")
    
    # Test analyze endpoint
    print("\n2. Testing Analyze Endpoint...")
    try:
        test_data = {"prompt": "Write a marketing email"}
        response = requests.post(
            f"{RAILWAY_URL}/api/v1/analyze", 
            json=test_data, 
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Analyze Endpoint: PASSED")
            print(f"   Clarity Score: {result.get('metrics', {}).get('clarity_score', 'N/A')}")
            print(f"   Suggestions: {len(result.get('suggestions', []))} found")
        else:
            print(f"❌ Analyze Endpoint: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Analyze Endpoint: FAILED (Error: {e})")
    
    # Test optimize endpoint
    print("\n3. Testing Optimize Endpoint...")
    try:
        test_data = {
            "prompt": "Write something good",
            "strategy": "comprehensive"
        }
        response = requests.post(
            f"{RAILWAY_URL}/api/v1/optimize", 
            json=test_data, 
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Optimize Endpoint: PASSED")
            print(f"   Original: {result.get('original_prompt', '')}")
            print(f"   Optimized: {result.get('optimized_prompt', '')[:100]}...")
            print(f"   Improvement: +{result.get('improvement_score', 0):.1f} points")
        else:
            print(f"❌ Optimize Endpoint: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Optimize Endpoint: FAILED (Error: {e})")
    
    print(f"\n🔗 Your API URL: {RAILWAY_URL}")
    print(f"📚 Documentation: {RAILWAY_URL}/docs")

if __name__ == "__main__":
    # Get URL from user
    url = input("Enter your Railway URL (or press Enter to use placeholder): ").strip()
    if url:
        RAILWAY_URL = url.rstrip('/')
    
    test_deployed_api()
