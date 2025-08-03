"""
Example usage of Perfect Prompt optimization system.
"""

import sys
import os

# Add the package to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perfect_prompt import PromptAnalyzer, PromptOptimizer, PerfectPromptClient


def basic_analysis_example():
    """Example of basic prompt analysis."""
    print("🔍 Basic Prompt Analysis Example")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PromptAnalyzer()
    
    # Example prompts
    prompts = [
        "Write something about AI",
        "Please write a comprehensive, detailed, and well-structured analysis of artificial intelligence technologies, including their current applications, future potential, and societal implications.",
        "Generate a 500-word essay about machine learning algorithms, focusing on neural networks and their applications in computer vision."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Prompt {i}: {prompt[:60]}...")
        
        # Analyze the prompt
        analysis = analyzer.analyze_prompt(prompt)
        
        # Display metrics
        metrics = analysis.metrics
        print(f"   Length: {metrics.length} characters")
        print(f"   Clarity Score: {metrics.clarity_score:.3f}")
        print(f"   Specificity Score: {metrics.specificity_score:.3f}")
        print(f"   Efficiency Score: {metrics.efficiency_score:.3f}")
        print(f"   Readability Score: {metrics.readability_score:.3f}")
        
        # Display suggestions
        if analysis.suggestions:
            print(f"   💡 Top Suggestion: {analysis.suggestions[0]}")


def optimization_example():
    """Example of prompt optimization."""
    print("\n🚀 Prompt Optimization Example")
    print("=" * 50)
    
    # Initialize optimizer
    analyzer = PromptAnalyzer()
    optimizer = PromptOptimizer(analyzer)
    
    # Example prompt to optimize
    original_prompt = "Write something good about machine learning"
    
    print(f"📝 Original: {original_prompt}")
    
    # Try different optimization strategies
    strategies = ["clarity_focused", "efficiency_focused", "comprehensive"]
    
    for strategy in strategies:
        print(f"\n🎯 Strategy: {strategy}")
        
        result = optimizer.optimize_prompt(original_prompt, strategy)
        
        print(f"   ✨ Optimized: {result.optimized_prompt}")
        print(f"   📈 Improvement: {result.improvement_score:.3f}")
        print(f"   🔧 Steps: {len(result.optimization_steps)}")


def comparison_example():
    """Example of prompt comparison."""
    print("\n⚖️ Prompt Comparison Example")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PromptAnalyzer()
    
    # Prompts to compare
    prompts = [
        "Explain AI",
        "Please provide a detailed explanation of artificial intelligence",
        "Write a comprehensive overview of AI technologies, including machine learning, deep learning, and their applications"
    ]
    
    print("📝 Comparing prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"   {i}. {prompt}")
    
    # Compare prompts
    comparison = analyzer.compare_prompts(prompts)
    
    # Display results
    best = comparison["best_overall"]
    print(f"\n🏆 Best Overall: Prompt #{best['index'] + 1}")
    print(f"   Score: {best['score']:.3f}")
    print(f"   Text: {best['prompt']}")
    
    # Show best in each metric
    print(f"\n📊 Best by Metric:")
    for metric_name, metric_data in comparison["best_metrics"].items():
        if metric_name in ["clarity_score", "specificity_score", "efficiency_score"]:
            print(f"   {metric_name}: Prompt #{metric_data['index'] + 1} ({metric_data['score']:.3f})")


def api_client_example():
    """Example of using the API client."""
    print("\n🌐 API Client Example")
    print("=" * 50)
    
    # Note: This requires the API server to be running
    print("📡 This example requires the Perfect Prompt API server to be running.")
    print("   Start the server with: python -m perfect_prompt.api.server")
    print("   Or use: uvicorn perfect_prompt.api.server:app --reload")
    
    # Uncomment the following to test with a running server:
    """
    client = PerfectPromptClient("http://localhost:8000")
    
    # Check if server is available
    if client.health_check():
        print("✅ API server is healthy")
        
        # Analyze a prompt
        response = client.analyze_prompt("Write about AI")
        
        if response.success:
            print(f"📊 Analysis successful")
            print(f"   Clarity: {response.metrics['clarity_score']:.3f}")
            print(f"   Suggestions: {len(response.suggestions)}")
        
        # Optimize a prompt
        response = client.optimize_prompt("Write about AI", "comprehensive")
        
        if response.success:
            print(f"🚀 Optimization successful")
            print(f"   Original: {response.original_prompt}")
            print(f"   Optimized: {response.optimized_prompt}")
    else:
        print("❌ API server is not available")
    
    client.close()
    """


def target_optimization_example():
    """Example of optimizing for specific target metrics."""
    print("\n🎯 Target Optimization Example")
    print("=" * 50)
    
    # Initialize optimizer
    analyzer = PromptAnalyzer()
    optimizer = PromptOptimizer(analyzer)
    
    original_prompt = "Do something with data"
    
    print(f"📝 Original: {original_prompt}")
    
    # Set target metrics
    target_metrics = {
        "clarity_score": 0.8,
        "specificity_score": 0.7,
        "efficiency_score": 0.6
    }
    
    print(f"🎯 Targets: {target_metrics}")
    
    # Optimize for targets
    result = optimizer.optimize_for_target(original_prompt, target_metrics)
    
    print(f"✨ Optimized: {result.optimized_prompt}")
    print(f"📈 Improvement: {result.improvement_score:.3f}")
    
    # Show final metrics
    final_metrics = result.final_analysis.metrics
    print(f"📊 Final Metrics:")
    print(f"   Clarity: {final_metrics.clarity_score:.3f} (target: {target_metrics['clarity_score']})")
    print(f"   Specificity: {final_metrics.specificity_score:.3f} (target: {target_metrics['specificity_score']})")
    print(f"   Efficiency: {final_metrics.efficiency_score:.3f} (target: {target_metrics['efficiency_score']})")


def integration_example():
    """Example of integrating Perfect Prompt into existing code."""
    print("\n🔗 Integration Example")
    print("=" * 50)
    
    # Simulate an existing AI model class
    class ExampleAIModel:
        def __init__(self, auto_optimize=True):
            # Add prompt optimization capability
            from perfect_prompt.api.client import PromptOptimizationMixin
            
            self.auto_optimize = auto_optimize
            if auto_optimize:
                self.prompt_client = None  # Would initialize in real scenario
        
        def generate_text(self, prompt):
            """Simulate text generation with prompt optimization."""
            if self.auto_optimize:
                print(f"   🔧 Original prompt: {prompt}")
                # In real scenario, would optimize here
                optimized_prompt = f"Please {prompt.lower()} with specific details and examples."
                print(f"   ✨ Optimized prompt: {optimized_prompt}")
                prompt = optimized_prompt
            
            # Simulate generation
            return f"Generated text based on: {prompt[:50]}..."
    
    # Use the integrated model
    model = ExampleAIModel(auto_optimize=True)
    result = model.generate_text("Write about machine learning")
    print(f"📄 Result: {result}")


if __name__ == "__main__":
    print("🎯 Perfect Prompt - Examples")
    print("=" * 60)
    
    try:
        # Run examples
        basic_analysis_example()
        optimization_example()
        comparison_example()
        target_optimization_example()
        integration_example()
        api_client_example()
        
        print("\n✅ All examples completed!")
        print("\n💡 Next steps:")
        print("   1. Start the API server: python -m perfect_prompt.api.server")
        print("   2. Try the web interface (if available)")
        print("   3. Integrate into your AI models")
        print("   4. Customize optimization strategies")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")
