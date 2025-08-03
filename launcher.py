"""
Perfect Prompt Launcher - Easy setup and demo script.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    # Map package names to their import names
    package_imports = {
        'spacy': 'spacy',
        'numpy': 'numpy', 
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    return missing_packages


def get_python_executable():
    """Get the correct Python executable (prefer venv if available)."""
    venv_python = Path("venv/Scripts/python.exe")
    if venv_python.exists():
        return str(venv_python.absolute())
    return sys.executable


def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    python_exe = get_python_executable()
    try:
        subprocess.check_call([
            python_exe, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def download_spacy_model():
    """Download the spaCy English model."""
    print("🔽 Downloading spaCy English model...")
    python_exe = get_python_executable()
    try:
        subprocess.check_call([
            python_exe, '-m', 'spacy', 'download', 'en_core_web_sm'
        ])
        print("✅ spaCy model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading spaCy model: {e}")
        print("💡 You can run the simplified demo without this model")
        return False


def run_simple_demo():
    """Run the simplified demo."""
    print("🎯 Running simplified Perfect Prompt demo...")
    python_exe = get_python_executable()
    try:
        subprocess.check_call([
            python_exe, 'examples/simple_demo.py'
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running demo: {e}")
        return False


def run_full_examples():
    """Run the full examples if dependencies are available."""
    print("🚀 Running full Perfect Prompt examples...")
    python_exe = get_python_executable()
    try:
        subprocess.check_call([
            python_exe, 'examples/basic_usage.py'
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running full examples: {e}")
        print("💡 Try running the simplified demo instead")
        return False


def start_api_server():
    """Start the FastAPI server."""
    print("🌐 Starting Perfect Prompt API server...")
    python_exe = get_python_executable()
    try:
        subprocess.Popen([
            python_exe, '-m', 'perfect_prompt.api.server'
        ])
        print("✅ API server started at http://localhost:8000")
        print("📚 API documentation available at http://localhost:8000/docs")
        return True
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return False


def main():
    """Main launcher function."""
    print("🎯 Perfect Prompt - AI-Powered Prompt Optimization")
    print("=" * 60)
    print()
    
    # Check current directory
    if not Path('perfect_prompt').exists():
        print("❌ Error: Please run this script from the Perfect Prompt project directory")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. 🎮 Run Simple Demo (no external dependencies)")
        print("2. 📦 Install Full Dependencies")
        print("3. 🔽 Download spaCy Model")
        print("4. 🚀 Run Full Examples")
        print("5. 🌐 Start API Server")
        print("6. 🧪 Run Tests")
        print("7. ℹ️  Show Project Info")
        print("8. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            run_simple_demo()
            
        elif choice == '2':
            print("\n" + "="*50)
            install_requirements()
            
        elif choice == '3':
            print("\n" + "="*50)
            download_spacy_model()
            
        elif choice == '4':
            print("\n" + "="*50)
            missing = check_requirements()
            if missing:
                print(f"❌ Missing packages: {', '.join(missing)}")
                print("💡 Run option 2 to install dependencies first")
            else:
                run_full_examples()
                
        elif choice == '5':
            print("\n" + "="*50)
            missing = check_requirements()
            if missing:
                print(f"❌ Missing packages: {', '.join(missing)}")
                print("💡 Run option 2 to install dependencies first")
            else:
                start_api_server()
                input("Press Enter to continue (server will keep running)...")
                
        elif choice == '6':
            print("\n" + "="*50)
            print("🧪 Running tests...")
            python_exe = get_python_executable()
            try:
                subprocess.check_call([
                    python_exe, '-m', 'pytest', 'tests/', '-v'
                ])
            except subprocess.CalledProcessError:
                print("❌ Some tests failed or pytest is not installed")
                
        elif choice == '7':
            print("\n" + "="*50)
            print("🎯 Perfect Prompt - Project Information")
            print("-" * 40)
            print("📖 Description: AI-powered prompt optimization system")
            print("🔗 Integration: Embeddable in other AI models")
            print("📊 Features: Analysis, optimization, comparison, API")
            print("🧠 ML Models: Random Forest, Gradient Boosting, Ensemble")
            print("🌐 API: FastAPI with REST endpoints")
            print("📚 Documentation: README.md and inline docs")
            print("🧪 Testing: pytest with comprehensive test suite")
            print("📦 Dependencies: spaCy, scikit-learn, FastAPI, and more")
            
        elif choice == '8':
            print("\n👋 Thanks for using Perfect Prompt!")
            print("🔗 GitHub: https://github.com/perfect-prompt/perfect-prompt")
            print("📧 Contact: team@perfectprompt.ai")
            break
            
        else:
            print("❌ Invalid choice. Please enter a number between 1-8.")


if __name__ == "__main__":
    main()
