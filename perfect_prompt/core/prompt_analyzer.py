"""
Prompt Analyzer: Core module for analyzing prompt effectiveness and characteristics.
"""

from typing import Dict, List, Optional, Any, Tuple
import re
import spacy
import numpy as np
from dataclasses import dataclass
from loguru import logger
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline


@dataclass
class PromptMetrics:
    """Metrics for prompt analysis."""
    
    length: int
    complexity_score: float
    clarity_score: float
    specificity_score: float
    sentiment_score: float
    readability_score: float
    semantic_density: float
    instruction_clarity: float
    context_richness: float
    efficiency_score: float


@dataclass
class PromptAnalysis:
    """Complete analysis result for a prompt."""
    
    prompt: str
    metrics: PromptMetrics
    suggestions: List[str]
    identified_patterns: List[str]
    potential_issues: List[str]
    optimization_opportunities: List[str]


class PromptAnalyzer:
    """
    Advanced prompt analyzer that evaluates prompt effectiveness
    and provides optimization suggestions.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the PromptAnalyzer.
        
        Args:
            model_name: spaCy model name for NLP processing
        """
        self.nlp = None
        self.sentiment_analyzer = None
        self.semantic_model = None
        self.model_name = model_name
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize NLP models and tools."""
        try:
            # Initialize spaCy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
            
            # Initialize NLTK sentiment analyzer
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize semantic similarity model (with fallback)
            try:
                self.semantic_model = pipeline(
                    "feature-extraction",
                    model="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Semantic model initialized successfully")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")
                logger.warning("Using fallback semantic analysis without transformers model")
                self.semantic_model = None
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Perform comprehensive analysis of a prompt.
        
        Args:
            prompt: The prompt text to analyze
            
        Returns:
            PromptAnalysis object with complete analysis results
        """
        logger.info(f"Analyzing prompt: {prompt[:50]}...")
        
        # Calculate all metrics
        metrics = self._calculate_metrics(prompt)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(prompt, metrics)
        
        # Identify patterns
        patterns = self._identify_patterns(prompt)
        
        # Find potential issues
        issues = self._find_potential_issues(prompt, metrics)
        
        # Find optimization opportunities
        optimizations = self._find_optimizations(prompt, metrics)
        
        return PromptAnalysis(
            prompt=prompt,
            metrics=metrics,
            suggestions=suggestions,
            identified_patterns=patterns,
            potential_issues=issues,
            optimization_opportunities=optimizations
        )
    
    def _calculate_metrics(self, prompt: str) -> PromptMetrics:
        """Calculate all prompt metrics."""
        doc = self.nlp(prompt)
        
        return PromptMetrics(
            length=len(prompt),
            complexity_score=self._calculate_complexity(doc),
            clarity_score=self._calculate_clarity(doc),
            specificity_score=self._calculate_specificity(doc),
            sentiment_score=self._calculate_sentiment(prompt),
            readability_score=self._calculate_readability(prompt),
            semantic_density=self._calculate_semantic_density(doc),
            instruction_clarity=self._calculate_instruction_clarity(doc),
            context_richness=self._calculate_context_richness(doc),
            efficiency_score=self._calculate_efficiency_score(prompt, doc)
        )
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate prompt complexity based on sentence structure."""
        if not doc.sents:
            return 0.0
        
        total_complexity = 0
        sentence_count = 0
        
        for sent in doc.sents:
            # Count subordinate clauses, dependencies
            clause_count = len([token for token in sent if token.dep_ in ["ccomp", "xcomp", "advcl"]])
            dependency_depth = max([len(list(token.ancestors)) for token in sent], default=0)
            
            complexity = (clause_count * 0.3) + (dependency_depth * 0.2) + (len(sent) * 0.01)
            total_complexity += complexity
            sentence_count += 1
        
        return min(total_complexity / sentence_count, 1.0) if sentence_count > 0 else 0.0
    
    def _calculate_clarity(self, doc) -> float:
        """Calculate how clear and unambiguous the prompt is."""
        # Check for clear action verbs
        action_verbs = ["write", "create", "generate", "analyze", "explain", "describe", "list"]
        has_action = any(token.lemma_.lower() in action_verbs for token in doc)
        
        # Check for ambiguous words
        ambiguous_words = ["thing", "stuff", "something", "anything", "maybe", "perhaps"]
        ambiguous_count = sum(1 for token in doc if token.lemma_.lower() in ambiguous_words)
        
        # Check sentence structure clarity
        clear_structure = len([sent for sent in doc.sents if len(sent) > 3]) / max(len(list(doc.sents)), 1)
        
        clarity = (
            (0.4 if has_action else 0.0) +
            (0.3 * (1 - min(ambiguous_count / len(doc), 0.5))) +
            (0.3 * clear_structure)
        )
        
        return min(clarity, 1.0)
    
    def _calculate_specificity(self, doc) -> float:
        """Calculate how specific the prompt is."""
        # Count specific entities, numbers, proper nouns
        entities = len(doc.ents)
        numbers = len([token for token in doc if token.like_num])
        proper_nouns = len([token for token in doc if token.pos_ == "PROPN"])
        
        # Check for specific adjectives and modifiers
        specific_modifiers = len([token for token in doc if token.pos_ in ["ADJ", "ADV"]])
        
        specificity = min(
            (entities * 0.2 + numbers * 0.3 + proper_nouns * 0.3 + specific_modifiers * 0.1) / len(doc),
            1.0
        )
        
        return specificity
    
    def _calculate_sentiment(self, prompt: str) -> float:
        """Calculate sentiment polarity of the prompt."""
        scores = self.sentiment_analyzer.polarity_scores(prompt)
        return scores['compound']
    
    def _calculate_readability(self, prompt: str) -> float:
        """Calculate readability score (simplified Flesch formula)."""
        sentences = len(re.findall(r'[.!?]+', prompt))
        words = len(prompt.split())
        syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in prompt.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        flesch_score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, flesch_score)) / 100.0
    
    def _calculate_semantic_density(self, doc) -> float:
        """Calculate semantic information density."""
        content_words = [token for token in doc if not token.is_stop and not token.is_punct]
        total_words = len([token for token in doc if not token.is_punct])
        
        if total_words == 0:
            return 0.0
        
        return len(content_words) / total_words
    
    def _calculate_instruction_clarity(self, doc) -> float:
        """Calculate how clear the instructions are."""
        # Look for imperative mood, clear commands
        imperatives = len([token for token in doc if token.tag_ == "VB"])
        questions = len(re.findall(r'\?', doc.text))
        
        # Check for step indicators
        step_indicators = len(re.findall(r'\b(first|second|then|next|finally|step)\b', doc.text.lower()))
        
        instruction_score = min(
            (imperatives * 0.4 + questions * 0.3 + step_indicators * 0.3) / len(doc),
            1.0
        )
        
        return instruction_score
    
    def _calculate_context_richness(self, doc) -> float:
        """Calculate how much context is provided."""
        # Count background information indicators
        context_words = ["because", "since", "given", "considering", "context", "background"]
        context_indicators = sum(1 for token in doc if token.lemma_.lower() in context_words)
        
        # Count examples and explanations
        example_indicators = len(re.findall(r'\b(example|instance|such as|like)\b', doc.text.lower()))
        
        context_score = min(
            (context_indicators * 0.5 + example_indicators * 0.5) / max(len(doc), 1),
            1.0
        )
        
        return context_score
    
    def _calculate_efficiency_score(self, prompt: str, doc) -> float:
        """Calculate overall prompt efficiency."""
        # Combine multiple factors for efficiency
        length_efficiency = 1.0 - min(len(prompt) / 1000, 0.5)  # Penalize very long prompts
        word_efficiency = len(set(token.lemma_.lower() for token in doc)) / len(doc)
        
        # Reward clear structure and purpose
        structure_score = len([sent for sent in doc.sents if len(sent) > 2]) / max(len(list(doc.sents)), 1)
        
        efficiency = (length_efficiency * 0.3 + word_efficiency * 0.4 + structure_score * 0.3)
        return min(efficiency, 1.0)
    
    def _generate_suggestions(self, prompt: str, metrics: PromptMetrics) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        if metrics.clarity_score < 0.5:
            suggestions.append("Consider using more specific action verbs and reducing ambiguous language")
        
        if metrics.specificity_score < 0.3:
            suggestions.append("Add more specific details, examples, or constraints to improve precision")
        
        if metrics.length > 500:
            suggestions.append("Consider breaking down into smaller, more focused prompts")
        
        if metrics.instruction_clarity < 0.4:
            suggestions.append("Use clearer imperatives and step-by-step instructions")
        
        if metrics.context_richness < 0.2:
            suggestions.append("Provide more background context or examples to guide the AI")
        
        if metrics.efficiency_score < 0.5:
            suggestions.append("Optimize for brevity while maintaining clarity and specificity")
        
        return suggestions
    
    def _identify_patterns(self, prompt: str) -> List[str]:
        """Identify common prompt patterns."""
        patterns = []
        
        if re.search(r'\b(act as|you are|imagine you are)\b', prompt.lower()):
            patterns.append("Role-playing pattern")
        
        if re.search(r'\b(step by step|numbered list|bullet points)\b', prompt.lower()):
            patterns.append("Structured output pattern")
        
        if re.search(r'\b(example|for instance|such as)\b', prompt.lower()):
            patterns.append("Example-driven pattern")
        
        if re.search(r'\b(format|structure|template)\b', prompt.lower()):
            patterns.append("Format specification pattern")
        
        return patterns
    
    def _find_potential_issues(self, prompt: str, metrics: PromptMetrics) -> List[str]:
        """Identify potential issues with the prompt."""
        issues = []
        
        if metrics.length < 10:
            issues.append("Prompt might be too short to provide sufficient guidance")
        
        if metrics.complexity_score > 0.8:
            issues.append("Prompt complexity might confuse the AI model")
        
        if metrics.sentiment_score < -0.5:
            issues.append("Negative sentiment might affect AI response quality")
        
        if "don't" in prompt.lower() or "avoid" in prompt.lower():
            issues.append("Negative instructions might be less effective than positive ones")
        
        return issues
    
    def _find_optimizations(self, prompt: str, metrics: PromptMetrics) -> List[str]:
        """Find optimization opportunities."""
        optimizations = []
        
        if metrics.efficiency_score < 0.6:
            optimizations.append("Optimize prompt length and word choice for better efficiency")
        
        if metrics.semantic_density < 0.5:
            optimizations.append("Increase information density by removing filler words")
        
        if not re.search(r'\b(please|kindly)\b', prompt.lower()):
            optimizations.append("Consider adding polite language for better AI cooperation")
        
        return optimizations
    
    def compare_prompts(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Compare multiple prompts and provide recommendations.
        
        Args:
            prompts: List of prompts to compare
            
        Returns:
            Comparison results with recommendations
        """
        analyses = [self.analyze_prompt(prompt) for prompt in prompts]
        
        # Find best performing prompt in each metric
        best_metrics = {}
        for metric_name in analyses[0].metrics.__dict__.keys():
            best_idx = max(
                range(len(analyses)),
                key=lambda i: getattr(analyses[i].metrics, metric_name)
            )
            best_metrics[metric_name] = {
                "index": best_idx,
                "prompt": prompts[best_idx],
                "score": getattr(analyses[best_idx].metrics, metric_name)
            }
        
        # Overall best prompt
        overall_scores = [
            analysis.metrics.efficiency_score for analysis in analyses
        ]
        best_overall_idx = max(range(len(overall_scores)), key=lambda i: overall_scores[i])
        
        return {
            "analyses": analyses,
            "best_metrics": best_metrics,
            "best_overall": {
                "index": best_overall_idx,
                "prompt": prompts[best_overall_idx],
                "score": overall_scores[best_overall_idx]
            },
            "recommendations": self._generate_comparison_recommendations(analyses)
        }
    
    def _generate_comparison_recommendations(self, analyses: List[PromptAnalysis]) -> List[str]:
        """Generate recommendations based on prompt comparison."""
        recommendations = []
        
        avg_clarity = np.mean([a.metrics.clarity_score for a in analyses])
        avg_specificity = np.mean([a.metrics.specificity_score for a in analyses])
        
        if avg_clarity < 0.5:
            recommendations.append("All prompts could benefit from improved clarity")
        
        if avg_specificity < 0.4:
            recommendations.append("Consider adding more specific details across all prompts")
        
        return recommendations
