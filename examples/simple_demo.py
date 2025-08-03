"""
Simplified Perfect Prompt example without external dependencies.
"""

import sys
import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SimplePromptMetrics:
    """Simplified metrics for prompt analysis."""
    
    length: int
    word_count: int
    sentence_count: int
    clarity_score: float
    specificity_score: float
    efficiency_score: float


@dataclass
class SimplePromptAnalysis:
    """Simplified analysis result for a prompt."""
    
    prompt: str
    metrics: SimplePromptMetrics
    suggestions: List[str]
    issues: List[str]


class SimplePromptAnalyzer:
    """
    Simplified prompt analyzer for demonstration purposes.
    """
    
    def analyze_prompt(self, prompt: str) -> SimplePromptAnalysis:
        """
        Perform basic analysis of a prompt.
        
        Args:
            prompt: The prompt text to analyze
            
        Returns:
            SimplePromptAnalysis object with results
        """
        # Calculate basic metrics
        metrics = self._calculate_metrics(prompt)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(prompt, metrics)
        
        # Find issues
        issues = self._find_issues(prompt, metrics)
        
        return SimplePromptAnalysis(
            prompt=prompt,
            metrics=metrics,
            suggestions=suggestions,
            issues=issues
        )
    
    def _calculate_metrics(self, prompt: str) -> SimplePromptMetrics:
        """Calculate simplified prompt metrics."""
        length = len(prompt)
        words = prompt.split()
        word_count = len(words)
        sentence_count = len(re.findall(r'[.!?]+', prompt))
        
        # Simple clarity score based on action words
        action_words = ['write', 'create', 'generate', 'analyze', 'explain', 'describe']
        has_action = any(word.lower() in action_words for word in words)
        clarity_score = 0.8 if has_action else 0.3
        
        # Simple specificity score based on details
        detail_indicators = ['specific', 'detailed', 'comprehensive', 'example']
        detail_count = sum(1 for word in words if word.lower() in detail_indicators)
        specificity_score = min(1.0, detail_count * 0.3)
        
        # Simple efficiency score (shorter is more efficient, but not too short)
        if length < 10:
            efficiency_score = 0.2
        elif length < 50:
            efficiency_score = 0.8
        elif length < 200:
            efficiency_score = 0.6
        else:
            efficiency_score = 0.4
        
        return SimplePromptMetrics(
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            clarity_score=clarity_score,
            specificity_score=specificity_score,
            efficiency_score=efficiency_score
        )
    
    def _generate_suggestions(self, prompt: str, metrics: SimplePromptMetrics) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if metrics.clarity_score < 0.5:
            suggestions.append("Add clear action words like 'write', 'create', or 'analyze'")
        
        if metrics.specificity_score < 0.3:
            suggestions.append("Include specific details, examples, or constraints")
        
        if metrics.length < 10:
            suggestions.append("Provide more context and detail")
        
        if metrics.length > 200:
            suggestions.append("Consider making the prompt more concise")
        
        if metrics.sentence_count == 0:
            suggestions.append("Structure the prompt with clear sentences")
        
        return suggestions
    
    def _find_issues(self, prompt: str, metrics: SimplePromptMetrics) -> List[str]:
        """Find potential issues with the prompt."""
        issues = []
        
        if metrics.length < 5:
            issues.append("Prompt is too short to be effective")
        
        if 'don\'t' in prompt.lower() or 'avoid' in prompt.lower():
            issues.append("Contains negative instructions which may be less effective")
        
        ambiguous_words = ['thing', 'stuff', 'something']
        if any(word in prompt.lower() for word in ambiguous_words):
            issues.append("Contains ambiguous language")
        
        return issues


class SimplePromptOptimizer:
    """
    Simplified prompt optimizer for demonstration.
    """
    
    def __init__(self):
        self.analyzer = SimplePromptAnalyzer()
    
    def optimize_prompt(self, prompt: str, strategy: str = "comprehensive") -> Dict:
        """
        Optimize a prompt using simple rules.
        
        Args:
            prompt: The prompt to optimize
            strategy: Optimization strategy (ignored in simple version)
            
        Returns:
            Dictionary with optimization results
        """
        original_analysis = self.analyzer.analyze_prompt(prompt)
        
        # Apply simple optimizations
        optimized_prompt = self._apply_optimizations(prompt, original_analysis)
        
        # Analyze optimized version
        final_analysis = self.analyzer.analyze_prompt(optimized_prompt)
        
        # Calculate improvement
        improvement = (
            final_analysis.metrics.clarity_score - original_analysis.metrics.clarity_score +
            final_analysis.metrics.specificity_score - original_analysis.metrics.specificity_score +
            final_analysis.metrics.efficiency_score - original_analysis.metrics.efficiency_score
        ) / 3
        
        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "improvement_score": improvement,
            "original_analysis": original_analysis,
            "final_analysis": final_analysis
        }
    
    def _apply_optimizations(self, prompt: str, analysis: SimplePromptAnalysis) -> str:
        """Apply simple optimization rules."""
        optimized = prompt
        
        # Add action word if missing
        if analysis.metrics.clarity_score < 0.5:
            if not any(word in prompt.lower() for word in ['write', 'create', 'generate']):
                optimized = f"Please write {prompt.lower()}"
        
        # Add specificity if needed
        if analysis.metrics.specificity_score < 0.3:
            if 'detailed' not in optimized.lower():
                optimized = optimized + " with specific details and examples"
        
        # Remove ambiguous words
        replacements = {
            'thing': 'item',
            'stuff': 'content',
            'something': 'a specific example'
        }
        
        for old, new in replacements.items():
            optimized = optimized.replace(old, new)
        
        return optimized


def demonstrate_simple_perfect_prompt():
    """Demonstrate the simplified Perfect Prompt functionality."""
    print("🎯 Perfect Prompt - Simplified Demo")
    print("=" * 50)
    
    # Initialize components
    analyzer = SimplePromptAnalyzer()
    optimizer = SimplePromptOptimizer()
    
    # Test prompts
    test_prompts = [
        "AI",
        "Write something about AI",
        "Create a comprehensive analysis of artificial intelligence technologies",
        "Don't write bad stuff about machine learning things"
    ]
    
    print("\n📊 Prompt Analysis Results:")
    print("-" * 30)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: \"{prompt}\"")
        
        analysis = analyzer.analyze_prompt(prompt)
        
        print(f"   📏 Length: {analysis.metrics.length} chars, {analysis.metrics.word_count} words")
        print(f"   📈 Clarity: {analysis.metrics.clarity_score:.2f}")
        print(f"   🎯 Specificity: {analysis.metrics.specificity_score:.2f}")
        print(f"   ⚡ Efficiency: {analysis.metrics.efficiency_score:.2f}")
        
        if analysis.suggestions:
            print(f"   💡 Suggestions: {', '.join(analysis.suggestions[:2])}")
        
        if analysis.issues:
            print(f"   ⚠️  Issues: {', '.join(analysis.issues[:2])}")
    
    print("\n🚀 Prompt Optimization Examples:")
    print("-" * 30)
    
    # Test optimization
    for i, prompt in enumerate(test_prompts[:3], 1):
        print(f"\n{i}. Optimizing: \"{prompt}\"")
        
        result = optimizer.optimize_prompt(prompt)
        
        print(f"   ✨ Optimized: \"{result['optimized_prompt']}\"")
        print(f"   📈 Improvement: {result['improvement_score']:.3f}")
        
        if result['improvement_score'] > 0.1:
            print("   ✅ Significant improvement achieved!")
        elif result['improvement_score'] > 0:
            print("   📊 Minor improvement achieved")
        else:
            print("   ℹ️  Prompt was already well-optimized")
    
    print("\n🔄 Comparison Example:")
    print("-" * 30)
    
    # Compare prompts
    comparison_prompts = [
        "Explain AI",
        "Write a detailed explanation of artificial intelligence",
        "Create a comprehensive analysis of AI technologies with examples"
    ]
    
    best_score = 0
    best_prompt = ""
    
    for i, prompt in enumerate(comparison_prompts, 1):
        analysis = analyzer.analyze_prompt(prompt)
        avg_score = (
            analysis.metrics.clarity_score + 
            analysis.metrics.specificity_score + 
            analysis.metrics.efficiency_score
        ) / 3
        
        print(f"{i}. \"{prompt}\" - Score: {avg_score:.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_prompt = prompt
    
    print(f"\n🏆 Best prompt: \"{best_prompt}\" (Score: {best_score:.3f})")
    
    print("\n💡 Key Features Demonstrated:")
    print("- ✅ Prompt analysis with multiple metrics")
    print("- ✅ Improvement suggestions")
    print("- ✅ Issue identification")
    print("- ✅ Automatic optimization")
    print("- ✅ Prompt comparison")
    
    print("\n🚀 Next Steps:")
    print("1. Install full dependencies: pip install -r requirements.txt")
    print("2. Download spaCy model: python -m spacy download en_core_web_sm")
    print("3. Run full version with advanced NLP features")
    print("4. Start API server for integration")
    print("5. Integrate into your AI applications")


if __name__ == "__main__":
    demonstrate_simple_perfect_prompt()
