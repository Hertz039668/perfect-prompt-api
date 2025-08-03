"""
Prompt Optimizer: Advanced optimization algorithms for improving prompt effectiveness.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from dataclasses import dataclass
from loguru import logger
from .prompt_analyzer import PromptAnalyzer, PromptAnalysis


@dataclass
class OptimizationStrategy:
    """Configuration for optimization strategy."""
    
    name: str
    description: str
    priority_weights: Dict[str, float]
    max_iterations: int = 10
    convergence_threshold: float = 0.01


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""
    
    original_prompt: str
    optimized_prompt: str
    improvement_score: float
    optimization_steps: List[str]
    final_analysis: PromptAnalysis
    strategy_used: str


class PromptOptimizer:
    """
    Advanced prompt optimizer that uses multiple strategies to improve
    prompt effectiveness and efficiency.
    """
    
    def __init__(self, analyzer: Optional[PromptAnalyzer] = None):
        """
        Initialize the PromptOptimizer.
        
        Args:
            analyzer: PromptAnalyzer instance to use for evaluation
        """
        self.analyzer = analyzer or PromptAnalyzer()
        self.strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize optimization strategies."""
        return {
            "clarity_focused": OptimizationStrategy(
                name="Clarity Focused",
                description="Optimize for maximum clarity and understanding",
                priority_weights={
                    "clarity_score": 0.4,
                    "instruction_clarity": 0.3,
                    "specificity_score": 0.2,
                    "efficiency_score": 0.1
                }
            ),
            "efficiency_focused": OptimizationStrategy(
                name="Efficiency Focused", 
                description="Optimize for brevity and efficiency",
                priority_weights={
                    "efficiency_score": 0.5,
                    "semantic_density": 0.3,
                    "clarity_score": 0.2
                }
            ),
            "comprehensive": OptimizationStrategy(
                name="Comprehensive",
                description="Balance all aspects of prompt quality",
                priority_weights={
                    "clarity_score": 0.2,
                    "specificity_score": 0.2,
                    "efficiency_score": 0.2,
                    "instruction_clarity": 0.15,
                    "context_richness": 0.15,
                    "readability_score": 0.1
                }
            ),
            "creative": OptimizationStrategy(
                name="Creative",
                description="Optimize for creative and engaging prompts",
                priority_weights={
                    "context_richness": 0.3,
                    "specificity_score": 0.25,
                    "clarity_score": 0.25,
                    "complexity_score": 0.2
                }
            )
        }
    
    def optimize_prompt(
        self, 
        prompt: str, 
        strategy: str = "comprehensive",
        target_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Optimize a prompt using the specified strategy.
        
        Args:
            prompt: The prompt to optimize
            strategy: Optimization strategy to use
            target_metrics: Target values for specific metrics
            
        Returns:
            OptimizationResult with the optimized prompt and details
        """
        logger.info(f"Optimizing prompt using {strategy} strategy")
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        strategy_config = self.strategies[strategy]
        original_analysis = self.analyzer.analyze_prompt(prompt)
        
        # Apply optimization techniques
        optimized_prompt = prompt
        optimization_steps = []
        
        # Iterative optimization
        for iteration in range(strategy_config.max_iterations):
            current_analysis = self.analyzer.analyze_prompt(optimized_prompt)
            
            # Calculate current score based on strategy weights
            current_score = self._calculate_weighted_score(
                current_analysis.metrics, 
                strategy_config.priority_weights
            )
            
            # Apply optimization techniques
            candidates = self._generate_optimization_candidates(
                optimized_prompt, 
                current_analysis, 
                strategy_config
            )
            
            # Evaluate candidates and select best
            best_candidate = self._select_best_candidate(
                candidates, 
                strategy_config.priority_weights
            )
            
            if best_candidate:
                previous_score = current_score
                new_analysis = self.analyzer.analyze_prompt(best_candidate)
                new_score = self._calculate_weighted_score(
                    new_analysis.metrics,
                    strategy_config.priority_weights
                )
                
                # Check for improvement
                if new_score > previous_score + strategy_config.convergence_threshold:
                    optimized_prompt = best_candidate
                    optimization_steps.append(
                        f"Iteration {iteration + 1}: Applied optimization (score: {new_score:.3f})"
                    )
                else:
                    optimization_steps.append(
                        f"Iteration {iteration + 1}: No significant improvement found, stopping"
                    )
                    break
            else:
                optimization_steps.append(
                    f"Iteration {iteration + 1}: No valid candidates generated"
                )
                break
        
        final_analysis = self.analyzer.analyze_prompt(optimized_prompt)
        
        # Calculate improvement
        original_score = self._calculate_weighted_score(
            original_analysis.metrics,
            strategy_config.priority_weights
        )
        final_score = self._calculate_weighted_score(
            final_analysis.metrics,
            strategy_config.priority_weights
        )
        
        improvement = final_score - original_score
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            improvement_score=improvement,
            optimization_steps=optimization_steps,
            final_analysis=final_analysis,
            strategy_used=strategy
        )
    
    def _calculate_weighted_score(
        self, 
        metrics, 
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score based on metrics and weights."""
        score = 0.0
        for metric_name, weight in weights.items():
            if hasattr(metrics, metric_name):
                score += getattr(metrics, metric_name) * weight
        return score
    
    def _generate_optimization_candidates(
        self,
        prompt: str,
        analysis: PromptAnalysis,
        strategy: OptimizationStrategy
    ) -> List[str]:
        """Generate candidate optimizations for a prompt."""
        candidates = []
        
        # Apply different optimization techniques
        candidates.extend(self._apply_clarity_optimizations(prompt, analysis))
        candidates.extend(self._apply_efficiency_optimizations(prompt, analysis))
        candidates.extend(self._apply_structure_optimizations(prompt, analysis))
        candidates.extend(self._apply_specificity_optimizations(prompt, analysis))
        
        return candidates
    
    def _apply_clarity_optimizations(self, prompt: str, analysis: PromptAnalysis) -> List[str]:
        """Apply clarity-focused optimizations."""
        candidates = []
        
        if analysis.metrics.clarity_score < 0.7:
            # Add clear action verbs
            action_starters = [
                "Please write", "Generate", "Create", "Analyze", "Explain", "Describe"
            ]
            
            for starter in action_starters:
                if not prompt.lower().startswith(starter.lower()):
                    candidates.append(f"{starter} {prompt.lower()}")
            
            # Remove ambiguous language
            ambiguous_replacements = {
                "thing": "item",
                "stuff": "content", 
                "something": "a specific example",
                "maybe": "",
                "perhaps": ""
            }
            
            modified_prompt = prompt
            for ambiguous, replacement in ambiguous_replacements.items():
                if ambiguous in modified_prompt.lower():
                    modified_prompt = modified_prompt.replace(ambiguous, replacement)
                    candidates.append(modified_prompt)
        
        return candidates
    
    def _apply_efficiency_optimizations(self, prompt: str, analysis: PromptAnalysis) -> List[str]:
        """Apply efficiency-focused optimizations."""
        candidates = []
        
        if analysis.metrics.length > 200:
            # Create more concise versions
            sentences = prompt.split('.')
            if len(sentences) > 2:
                # Try keeping only the most important sentences
                for i in range(1, len(sentences)):
                    condensed = '. '.join(sentences[:i]) + '.'
                    if len(condensed) > 50:  # Ensure minimum length
                        candidates.append(condensed)
        
        # Remove redundant words
        redundant_phrases = [
            "please note that", "it should be noted that", "you should",
            "make sure to", "be sure to", "in order to"
        ]
        
        for phrase in redundant_phrases:
            if phrase in prompt.lower():
                simplified = prompt.replace(phrase, "").strip()
                candidates.append(simplified)
        
        return candidates
    
    def _apply_structure_optimizations(self, prompt: str, analysis: PromptAnalysis) -> List[str]:
        """Apply structure-focused optimizations."""
        candidates = []
        
        if analysis.metrics.instruction_clarity < 0.6:
            # Add structure indicators
            if "step" not in prompt.lower():
                structured_versions = [
                    f"Step by step: {prompt}",
                    f"Please follow these steps:\n1. {prompt}",
                    f"Instructions:\n- {prompt}"
                ]
                candidates.extend(structured_versions)
        
        return candidates
    
    def _apply_specificity_optimizations(self, prompt: str, analysis: PromptAnalysis) -> List[str]:
        """Apply specificity-focused optimizations."""
        candidates = []
        
        if analysis.metrics.specificity_score < 0.5:
            # Add specificity enhancers
            specificity_additions = [
                "with specific examples",
                "in detail", 
                "with concrete details",
                "including relevant context"
            ]
            
            for addition in specificity_additions:
                if addition not in prompt.lower():
                    candidates.append(f"{prompt} {addition}")
        
        return candidates
    
    def _select_best_candidate(
        self, 
        candidates: List[str], 
        weights: Dict[str, float]
    ) -> Optional[str]:
        """Select the best candidate from the list."""
        if not candidates:
            return None
        
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            try:
                analysis = self.analyzer.analyze_prompt(candidate)
                score = self._calculate_weighted_score(analysis.metrics, weights)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    
            except Exception as e:
                logger.warning(f"Error analyzing candidate: {e}")
                continue
        
        return best_candidate
    
    def optimize_for_target(
        self,
        prompt: str,
        target_metrics: Dict[str, float],
        max_iterations: int = 15
    ) -> OptimizationResult:
        """
        Optimize a prompt to achieve specific target metrics.
        
        Args:
            prompt: The prompt to optimize
            target_metrics: Dictionary of metric names and target values
            max_iterations: Maximum optimization iterations
            
        Returns:
            OptimizationResult with optimization details
        """
        logger.info(f"Optimizing prompt for target metrics: {target_metrics}")
        
        optimized_prompt = prompt
        optimization_steps = []
        
        for iteration in range(max_iterations):
            current_analysis = self.analyzer.analyze_prompt(optimized_prompt)
            
            # Calculate distance from targets
            distances = {}
            for metric_name, target_value in target_metrics.items():
                if hasattr(current_analysis.metrics, metric_name):
                    current_value = getattr(current_analysis.metrics, metric_name)
                    distances[metric_name] = abs(target_value - current_value)
            
            # Find metric with largest distance
            if distances:
                worst_metric = max(distances, key=distances.get)
                
                # Apply targeted optimization for worst metric
                candidates = self._generate_targeted_candidates(
                    optimized_prompt, 
                    worst_metric, 
                    target_metrics[worst_metric]
                )
                
                if candidates:
                    # Select candidate that best improves the target metric
                    best_candidate = self._select_candidate_for_metric(
                        candidates, 
                        worst_metric, 
                        target_metrics[worst_metric]
                    )
                    
                    if best_candidate:
                        optimized_prompt = best_candidate
                        optimization_steps.append(
                            f"Iteration {iteration + 1}: Improved {worst_metric} targeting {target_metrics[worst_metric]}"
                        )
                    else:
                        break
                else:
                    break
            else:
                break
        
        final_analysis = self.analyzer.analyze_prompt(optimized_prompt)
        
        # Calculate overall improvement
        original_analysis = self.analyzer.analyze_prompt(prompt)
        improvement = self._calculate_target_improvement(
            original_analysis.metrics,
            final_analysis.metrics,
            target_metrics
        )
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            improvement_score=improvement,
            optimization_steps=optimization_steps,
            final_analysis=final_analysis,
            strategy_used="target_optimization"
        )
    
    def _generate_targeted_candidates(
        self, 
        prompt: str, 
        metric: str, 
        target_value: float
    ) -> List[str]:
        """Generate candidates specifically for improving a target metric."""
        candidates = []
        
        if metric == "clarity_score":
            candidates.extend(self._generate_clarity_candidates(prompt))
        elif metric == "specificity_score":
            candidates.extend(self._generate_specificity_candidates(prompt))
        elif metric == "efficiency_score":
            candidates.extend(self._generate_efficiency_candidates(prompt))
        elif metric == "readability_score":
            candidates.extend(self._generate_readability_candidates(prompt))
        
        return candidates
    
    def _generate_clarity_candidates(self, prompt: str) -> List[str]:
        """Generate candidates to improve clarity."""
        return [
            f"Clear instruction: {prompt}",
            f"Please {prompt.lower()}",
            prompt.replace("you should", "please"),
            prompt.replace("might", "will")
        ]
    
    def _generate_specificity_candidates(self, prompt: str) -> List[str]:
        """Generate candidates to improve specificity."""
        return [
            f"{prompt} Be specific and detailed.",
            f"{prompt} Include concrete examples.",
            f"{prompt} Provide step-by-step details."
        ]
    
    def _generate_efficiency_candidates(self, prompt: str) -> List[str]:
        """Generate candidates to improve efficiency."""
        # Remove filler words and phrases
        efficiency_candidates = []
        
        filler_words = ["really", "very", "quite", "rather", "just", "actually"]
        modified = prompt
        
        for filler in filler_words:
            modified = modified.replace(f" {filler} ", " ")
        
        efficiency_candidates.append(modified)
        
        # Create more direct versions
        if "could you" in prompt.lower():
            efficiency_candidates.append(prompt.replace("Could you", ""))
        
        return efficiency_candidates
    
    def _generate_readability_candidates(self, prompt: str) -> List[str]:
        """Generate candidates to improve readability."""
        return [
            prompt.replace(".", ". "),  # Ensure proper spacing
            prompt.replace(",", ", "),  # Ensure proper spacing
            prompt.replace(";", ". "),  # Break complex sentences
        ]
    
    def _select_candidate_for_metric(
        self, 
        candidates: List[str], 
        metric: str, 
        target_value: float
    ) -> Optional[str]:
        """Select candidate that best improves the specific metric."""
        best_candidate = None
        best_distance = float('inf')
        
        for candidate in candidates:
            try:
                analysis = self.analyzer.analyze_prompt(candidate)
                if hasattr(analysis.metrics, metric):
                    current_value = getattr(analysis.metrics, metric)
                    distance = abs(target_value - current_value)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_candidate = candidate
            except Exception as e:
                logger.warning(f"Error analyzing candidate for metric {metric}: {e}")
                continue
        
        return best_candidate
    
    def _calculate_target_improvement(
        self, 
        original_metrics, 
        final_metrics, 
        targets: Dict[str, float]
    ) -> float:
        """Calculate improvement towards target metrics."""
        total_improvement = 0.0
        metric_count = 0
        
        for metric_name, target_value in targets.items():
            if hasattr(original_metrics, metric_name) and hasattr(final_metrics, metric_name):
                original_distance = abs(target_value - getattr(original_metrics, metric_name))
                final_distance = abs(target_value - getattr(final_metrics, metric_name))
                
                improvement = original_distance - final_distance
                total_improvement += improvement
                metric_count += 1
        
        return total_improvement / metric_count if metric_count > 0 else 0.0
    
    def batch_optimize(
        self, 
        prompts: List[str], 
        strategy: str = "comprehensive"
    ) -> List[OptimizationResult]:
        """
        Optimize multiple prompts using the same strategy.
        
        Args:
            prompts: List of prompts to optimize
            strategy: Optimization strategy to use
            
        Returns:
            List of OptimizationResult objects
        """
        logger.info(f"Batch optimizing {len(prompts)} prompts")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Optimizing prompt {i + 1}/{len(prompts)}")
            result = self.optimize_prompt(prompt, strategy)
            results.append(result)
        
        return results
