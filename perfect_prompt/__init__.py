"""
Perfect Prompt: AI-powered prompt optimization system.

This package provides tools for analyzing, optimizing, and suggesting
the most efficient prompts for AI models.
"""

__version__ = "0.1.0"
__author__ = "Perfect Prompt Team"
__email__ = "team@perfectprompt.ai"

from .core.prompt_analyzer import PromptAnalyzer
from .core.prompt_optimizer import PromptOptimizer
from .models.prompt_model import PromptModel

# Only import client if aiohttp is available (optional dependency)
try:
    from .api.client import PerfectPromptClient
except ImportError:
    # Client not available in production environment
    PerfectPromptClient = None

__all__ = [
    "PromptAnalyzer",
    "PromptOptimizer", 
    "PromptModel",
    "PerfectPromptClient",
]
