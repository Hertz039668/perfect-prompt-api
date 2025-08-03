"""
Tests for PromptAnalyzer functionality.
"""

import pytest
import sys
import os

# Add the package to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perfect_prompt.core.prompt_analyzer import PromptAnalyzer, PromptMetrics, PromptAnalysis


class TestPromptAnalyzer:
    """Test cases for PromptAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a PromptAnalyzer instance for testing."""
        return PromptAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.nlp is not None
        assert analyzer.sentiment_analyzer is not None
    
    def test_basic_analysis(self, analyzer):
        """Test basic prompt analysis."""
        prompt = "Write a detailed essay about machine learning algorithms."
        
        analysis = analyzer.analyze_prompt(prompt)
        
        assert isinstance(analysis, PromptAnalysis)
        assert analysis.prompt == prompt
        assert isinstance(analysis.metrics, PromptMetrics)
        assert isinstance(analysis.suggestions, list)
        assert isinstance(analysis.identified_patterns, list)
        assert isinstance(analysis.potential_issues, list)
    
    def test_metrics_calculation(self, analyzer):
        """Test that metrics are calculated within expected ranges."""
        prompt = "Please write a comprehensive analysis of artificial intelligence."
        
        analysis = analyzer.analyze_prompt(prompt)
        metrics = analysis.metrics
        
        # Check that metrics are within valid ranges
        assert 0 <= metrics.clarity_score <= 1
        assert 0 <= metrics.specificity_score <= 1
        assert 0 <= metrics.efficiency_score <= 1
        assert 0 <= metrics.complexity_score <= 1
        assert -1 <= metrics.sentiment_score <= 1
        assert 0 <= metrics.readability_score <= 1
        assert 0 <= metrics.semantic_density <= 1
        assert 0 <= metrics.instruction_clarity <= 1
        assert 0 <= metrics.context_richness <= 1
        
        # Check basic expectations
        assert metrics.length == len(prompt)
        assert metrics.length > 0
    
    def test_short_prompt_analysis(self, analyzer):
        """Test analysis of very short prompts."""
        prompt = "AI"
        
        analysis = analyzer.analyze_prompt(prompt)
        
        assert analysis.prompt == prompt
        assert analysis.metrics.length == 2
        # Short prompts should have potential issues
        assert any("short" in issue.lower() for issue in analysis.potential_issues)
    
    def test_long_prompt_analysis(self, analyzer):
        """Test analysis of very long prompts."""
        prompt = "Write " * 100 + "about artificial intelligence and machine learning."
        
        analysis = analyzer.analyze_prompt(prompt)
        
        assert analysis.prompt == prompt
        assert analysis.metrics.length > 500
        # Long prompts should have suggestions
        assert len(analysis.suggestions) > 0
    
    def test_clear_prompt_analysis(self, analyzer):
        """Test analysis of clear, well-structured prompts."""
        prompt = "Please write a 500-word essay analyzing the impact of machine learning on healthcare, including specific examples and potential future developments."
        
        analysis = analyzer.analyze_prompt(prompt)
        
        # Clear prompts should have good clarity scores
        assert analysis.metrics.clarity_score > 0.5
        assert analysis.metrics.specificity_score > 0.3
    
    def test_ambiguous_prompt_analysis(self, analyzer):
        """Test analysis of ambiguous prompts."""
        prompt = "Write something about stuff and things maybe."
        
        analysis = analyzer.analyze_prompt(prompt)
        
        # Ambiguous prompts should have lower clarity scores
        assert analysis.metrics.clarity_score < 0.7
        # Should have suggestions for improvement
        assert len(analysis.suggestions) > 0
    
    def test_compare_prompts(self, analyzer):
        """Test prompt comparison functionality."""
        prompts = [
            "AI",
            "Write about AI",
            "Please write a detailed analysis of artificial intelligence technologies"
        ]
        
        comparison = analyzer.compare_prompts(prompts)
        
        assert "analyses" in comparison
        assert "best_overall" in comparison
        assert "best_metrics" in comparison
        assert "recommendations" in comparison
        
        assert len(comparison["analyses"]) == 3
        assert 0 <= comparison["best_overall"]["index"] < 3
        assert comparison["best_overall"]["prompt"] in prompts
    
    def test_pattern_identification(self, analyzer):
        """Test identification of prompt patterns."""
        test_cases = [
            ("Act as a professional writer and create content", ["Role-playing pattern"]),
            ("Write step by step instructions", ["Structured output pattern"]),
            ("For example, consider this case", ["Example-driven pattern"]),
            ("Format the output as JSON", ["Format specification pattern"])
        ]
        
        for prompt, expected_patterns in test_cases:
            analysis = analyzer.analyze_prompt(prompt)
            
            for pattern in expected_patterns:
                assert any(pattern.lower() in p.lower() for p in analysis.identified_patterns), \
                    f"Expected pattern '{pattern}' not found in {analysis.identified_patterns}"
    
    def test_negative_sentiment_detection(self, analyzer):
        """Test detection of negative sentiment in prompts."""
        prompt = "Don't write terrible content that nobody wants to read"
        
        analysis = analyzer.analyze_prompt(prompt)
        
        # Should detect negative sentiment
        assert analysis.metrics.sentiment_score < 0
        # Should identify as potential issue
        assert any("negative" in issue.lower() for issue in analysis.potential_issues)
    
    def test_instruction_clarity(self, analyzer):
        """Test instruction clarity scoring."""
        clear_prompt = "Please write a 300-word summary of the main benefits of renewable energy."
        unclear_prompt = "Maybe you could possibly write something about energy or whatever."
        
        clear_analysis = analyzer.analyze_prompt(clear_prompt)
        unclear_analysis = analyzer.analyze_prompt(unclear_prompt)
        
        assert clear_analysis.metrics.instruction_clarity > unclear_analysis.metrics.instruction_clarity
    
    def test_context_richness(self, analyzer):
        """Test context richness scoring."""
        rich_prompt = "Given the current climate crisis context, write an analysis of renewable energy solutions, providing specific examples and considering economic factors."
        poor_prompt = "Write about energy."
        
        rich_analysis = analyzer.analyze_prompt(rich_prompt)
        poor_analysis = analyzer.analyze_prompt(poor_prompt)
        
        assert rich_analysis.metrics.context_richness > poor_analysis.metrics.context_richness
    
    def test_efficiency_scoring(self, analyzer):
        """Test efficiency scoring."""
        efficient_prompt = "Summarize AI benefits in 100 words."
        inefficient_prompt = "Well, I was thinking that maybe you could possibly consider writing, if you don't mind, a sort of summary about artificial intelligence and its various benefits and advantages."
        
        efficient_analysis = analyzer.analyze_prompt(efficient_prompt)
        inefficient_analysis = analyzer.analyze_prompt(inefficient_prompt)
        
        assert efficient_analysis.metrics.efficiency_score > inefficient_analysis.metrics.efficiency_score
    
    def test_empty_prompt_handling(self, analyzer):
        """Test handling of empty or whitespace-only prompts."""
        empty_prompts = ["", "   ", "\n\t"]
        
        for prompt in empty_prompts:
            analysis = analyzer.analyze_prompt(prompt)
            
            assert analysis.prompt == prompt
            assert analysis.metrics.length <= len(prompt)
            # Should have issues identified
            assert len(analysis.potential_issues) > 0
    
    def test_special_characters_handling(self, analyzer):
        """Test handling of prompts with special characters."""
        prompt = "Write about AI! #MachineLearning @2024 $$$"
        
        analysis = analyzer.analyze_prompt(prompt)
        
        assert analysis.prompt == prompt
        assert analysis.metrics.length == len(prompt)
        # Should still provide meaningful analysis
        assert analysis.metrics.clarity_score >= 0


if __name__ == "__main__":
    pytest.main([__file__])
