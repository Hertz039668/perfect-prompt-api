"""
Perfect Prompt API Deployment Checker
Verifies that your deployed API is working correctly.
"""

import requests
import json
import sys
from urllib.parse import urlparse

def check_api_health(base_url: str) -> bool:
    """Check if the API is responding."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            return True
        else:
            print(f"❌ API Health Check: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API Health Check: FAILED (Error: {e})")
        return False

def check_cors_headers(base_url: str) -> bool:
    """Check if CORS headers are properly configured."""
    try:
        response = requests.options(f"{base_url}/api/v1/analyze", 
                                  headers={'Origin': 'https://chat.openai.com'}, 
                                  timeout=10)
        cors_header = response.headers.get('Access-Control-Allow-Origin')
        if cors_header:
            print("✅ CORS Configuration: PASSED")
            return True
        else:
            print("❌ CORS Configuration: FAILED (Missing headers)")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ CORS Configuration: FAILED (Error: {e})")
        return False

def test_analyze_endpoint(base_url: str) -> bool:
    """Test the analyze endpoint with a sample prompt."""
    try:
        test_data = {
            "prompt": "Write a blog post about AI technology"
        }
        response = requests.post(f"{base_url}/api/v1/analyze", 
                               json=test_data, 
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Analyze Endpoint: PASSED")
                return True
            else:
                print(f"❌ Analyze Endpoint: FAILED (API Error: {result.get('error_message')})")
                return False
        else:
            print(f"❌ Analyze Endpoint: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Analyze Endpoint: FAILED (Error: {e})")
        return False

def test_optimize_endpoint(base_url: str) -> bool:
    """Test the optimize endpoint with a sample prompt."""
    try:
        test_data = {
            "prompt": "Write something good",
            "strategy": "comprehensive"
        }
        response = requests.post(f"{base_url}/api/v1/optimize", 
                               json=test_data, 
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Optimize Endpoint: PASSED")
                return True
            else:
                print(f"❌ Optimize Endpoint: FAILED (API Error: {result.get('error_message')})")
                return False
        else:
            print(f"❌ Optimize Endpoint: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Optimize Endpoint: FAILED (Error: {e})")
        return False

def check_documentation(base_url: str) -> bool:
    """Check if API documentation is accessible."""
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API Documentation: ACCESSIBLE")
            return True
        else:
            print(f"❌ API Documentation: NOT ACCESSIBLE (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API Documentation: NOT ACCESSIBLE (Error: {e})")
        return False

def main():
    print("🚀 Perfect Prompt API Deployment Checker")
    print("=" * 50)
    
    # Get API URL from user
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = input("Enter your deployed API URL (e.g., https://your-app.railway.app): ").strip()
    
    # Clean up URL
    if not api_url.startswith(('http://', 'https://')):
        api_url = 'https://' + api_url
    
    # Remove trailing slash
    api_url = api_url.rstrip('/')
    
    print(f"\nTesting API at: {api_url}")
    print("-" * 50)
    
    # Run all checks
    checks = [
        ("Health Check", lambda: check_api_health(api_url)),
        ("CORS Configuration", lambda: check_cors_headers(api_url)),
        ("Analyze Endpoint", lambda: test_analyze_endpoint(api_url)),
        ("Optimize Endpoint", lambda: test_optimize_endpoint(api_url)),
        ("Documentation", lambda: check_documentation(api_url))
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 Running {check_name}...")
        results.append(check_func())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT CHECK SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("\n✅ Your API is ready for ChatGPT integration!")
        print(f"\n📋 Next Steps:")
        print(f"1. Copy this URL for ChatGPT: {api_url}")
        print(f"2. Use the schema from: deploy/chatgpt-schema.json")
        print(f"3. Follow the ChatGPT integration guide in DEPLOYMENT_GUIDE.md")
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total})")
        print(f"\n🔧 Please fix the failing checks before proceeding with ChatGPT integration.")
        print(f"📖 Check the deployment logs and troubleshooting guide.")
    
    print(f"\n🔗 API Documentation: {api_url}/docs")
    print(f"🏥 Health Endpoint: {api_url}/health")

if __name__ == "__main__":
    main()
