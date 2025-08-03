"""
Perfect Prompt API Client: Client library for integrating with Perfect Prompt services.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import aiohttp
import requests
from dataclasses import dataclass, asdict
from loguru import logger
import json
from urllib.parse import urljoin


@dataclass
class PromptRequest:
    """Request object for prompt analysis/optimization."""
    
    prompt: str
    strategy: str = "comprehensive"
    target_metrics: Optional[Dict[str, float]] = None
    include_suggestions: bool = True
    include_analysis: bool = True


@dataclass
class PromptResponse:
    """Response object from Perfect Prompt API."""
    
    success: bool
    original_prompt: str
    optimized_prompt: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


class PerfectPromptClient:
    """
    Client for integrating Perfect Prompt optimization into other AI models and applications.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the Perfect Prompt client.
        
        Args:
            base_url: Base URL of the Perfect Prompt API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = None
        
        # Default headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "PerfectPrompt-Client/0.1.0"
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _get_session(self) -> requests.Session:
        """Get or create requests session."""
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update(self.headers)
        return self.session
    
    def analyze_prompt(self, prompt: str) -> PromptResponse:
        """
        Analyze a prompt for effectiveness metrics.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            PromptResponse with analysis results
        """
        try:
            session = self._get_session()
            url = urljoin(self.base_url, "/api/v1/analyze")
            
            payload = {
                "prompt": prompt,
                "include_suggestions": True
            }
            
            response = session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            return PromptResponse(
                success=True,
                original_prompt=prompt,
                analysis=data.get("analysis"),
                suggestions=data.get("suggestions"),
                metrics=data.get("metrics")
            )
            
        except requests.RequestException as e:
            logger.error(f"Error analyzing prompt: {e}")
            return PromptResponse(
                success=False,
                original_prompt=prompt,
                error_message=str(e)
            )
    
    def optimize_prompt(
        self, 
        prompt: str, 
        strategy: str = "comprehensive",
        target_metrics: Optional[Dict[str, float]] = None
    ) -> PromptResponse:
        """
        Optimize a prompt for better effectiveness.
        
        Args:
            prompt: The prompt to optimize
            strategy: Optimization strategy to use
            target_metrics: Target values for specific metrics
            
        Returns:
            PromptResponse with optimized prompt
        """
        try:
            session = self._get_session()
            url = urljoin(self.base_url, "/api/v1/optimize")
            
            payload = {
                "prompt": prompt,
                "strategy": strategy,
                "target_metrics": target_metrics,
                "include_analysis": True
            }
            
            response = session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            return PromptResponse(
                success=True,
                original_prompt=prompt,
                optimized_prompt=data.get("optimized_prompt"),
                analysis=data.get("analysis"),
                suggestions=data.get("suggestions"),
                metrics=data.get("metrics")
            )
            
        except requests.RequestException as e:
            logger.error(f"Error optimizing prompt: {e}")
            return PromptResponse(
                success=False,
                original_prompt=prompt,
                error_message=str(e)
            )
    
    def batch_optimize(
        self, 
        prompts: List[str], 
        strategy: str = "comprehensive"
    ) -> List[PromptResponse]:
        """
        Optimize multiple prompts in batch.
        
        Args:
            prompts: List of prompts to optimize
            strategy: Optimization strategy to use
            
        Returns:
            List of PromptResponse objects
        """
        try:
            session = self._get_session()
            url = urljoin(self.base_url, "/api/v1/batch-optimize")
            
            payload = {
                "prompts": prompts,
                "strategy": strategy
            }
            
            response = session.post(
                url, 
                json=payload, 
                timeout=self.timeout * 2  # Longer timeout for batch
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for i, result in enumerate(data.get("results", [])):
                results.append(PromptResponse(
                    success=True,
                    original_prompt=prompts[i],
                    optimized_prompt=result.get("optimized_prompt"),
                    analysis=result.get("analysis"),
                    suggestions=result.get("suggestions"),
                    metrics=result.get("metrics")
                ))
            
            return results
            
        except requests.RequestException as e:
            logger.error(f"Error in batch optimization: {e}")
            # Return error responses for all prompts
            return [
                PromptResponse(
                    success=False,
                    original_prompt=prompt,
                    error_message=str(e)
                ) for prompt in prompts
            ]
    
    def compare_prompts(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Compare multiple prompts and get recommendations.
        
        Args:
            prompts: List of prompts to compare
            
        Returns:
            Comparison results
        """
        try:
            session = self._get_session()
            url = urljoin(self.base_url, "/api/v1/compare")
            
            payload = {"prompts": prompts}
            
            response = session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error comparing prompts: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }
    
    def get_suggestions(self, prompt: str) -> List[str]:
        """
        Get improvement suggestions for a prompt.
        
        Args:
            prompt: The prompt to get suggestions for
            
        Returns:
            List of improvement suggestions
        """
        response = self.analyze_prompt(prompt)
        if response.success and response.suggestions:
            return response.suggestions
        return []
    
    def health_check(self) -> bool:
        """
        Check if the Perfect Prompt API is healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            session = self._get_session()
            url = urljoin(self.base_url, "/health")
            
            response = session.get(url, timeout=5)
            return response.status_code == 200
            
        except requests.RequestException:
            return False
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information and capabilities.
        
        Returns:
            API information dictionary
        """
        try:
            session = self._get_session()
            url = urljoin(self.base_url, "/api/v1/info")
            
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error getting API info: {e}")
            return {"error": str(e)}
    
    def close(self) -> None:
        """Close the client session."""
        if self.session:
            self.session.close()
            self.session = None


class AsyncPerfectPromptClient:
    """
    Async client for integrating Perfect Prompt optimization into other AI models.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the async Perfect Prompt client.
        
        Args:
            base_url: Base URL of the Perfect Prompt API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Default headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "PerfectPrompt-AsyncClient/0.1.0"
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def analyze_prompt(self, prompt: str) -> PromptResponse:
        """
        Analyze a prompt for effectiveness metrics (async).
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            PromptResponse with analysis results
        """
        try:
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                url = urljoin(self.base_url, "/api/v1/analyze")
                
                payload = {
                    "prompt": prompt,
                    "include_suggestions": True
                }
                
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return PromptResponse(
                        success=True,
                        original_prompt=prompt,
                        analysis=data.get("analysis"),
                        suggestions=data.get("suggestions"),
                        metrics=data.get("metrics")
                    )
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error analyzing prompt: {e}")
            return PromptResponse(
                success=False,
                original_prompt=prompt,
                error_message=str(e)
            )
    
    async def optimize_prompt(
        self, 
        prompt: str, 
        strategy: str = "comprehensive",
        target_metrics: Optional[Dict[str, float]] = None
    ) -> PromptResponse:
        """
        Optimize a prompt for better effectiveness (async).
        
        Args:
            prompt: The prompt to optimize
            strategy: Optimization strategy to use
            target_metrics: Target values for specific metrics
            
        Returns:
            PromptResponse with optimized prompt
        """
        try:
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                url = urljoin(self.base_url, "/api/v1/optimize")
                
                payload = {
                    "prompt": prompt,
                    "strategy": strategy,
                    "target_metrics": target_metrics,
                    "include_analysis": True
                }
                
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return PromptResponse(
                        success=True,
                        original_prompt=prompt,
                        optimized_prompt=data.get("optimized_prompt"),
                        analysis=data.get("analysis"),
                        suggestions=data.get("suggestions"),
                        metrics=data.get("metrics")
                    )
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error optimizing prompt: {e}")
            return PromptResponse(
                success=False,
                original_prompt=prompt,
                error_message=str(e)
            )
    
    async def batch_optimize(
        self, 
        prompts: List[str], 
        strategy: str = "comprehensive"
    ) -> List[PromptResponse]:
        """
        Optimize multiple prompts in batch (async).
        
        Args:
            prompts: List of prompts to optimize
            strategy: Optimization strategy to use
            
        Returns:
            List of PromptResponse objects
        """
        try:
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout * 2)
            ) as session:
                url = urljoin(self.base_url, "/api/v1/batch-optimize")
                
                payload = {
                    "prompts": prompts,
                    "strategy": strategy
                }
                
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    results = []
                    for i, result in enumerate(data.get("results", [])):
                        results.append(PromptResponse(
                            success=True,
                            original_prompt=prompts[i],
                            optimized_prompt=result.get("optimized_prompt"),
                            analysis=result.get("analysis"),
                            suggestions=result.get("suggestions"),
                            metrics=result.get("metrics")
                        ))
                    
                    return results
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error in batch optimization: {e}")
            return [
                PromptResponse(
                    success=False,
                    original_prompt=prompt,
                    error_message=str(e)
                ) for prompt in prompts
            ]
    
    async def health_check(self) -> bool:
        """
        Check if the Perfect Prompt API is healthy (async).
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                url = urljoin(self.base_url, "/health")
                
                async with session.get(url) as response:
                    return response.status == 200
                    
        except aiohttp.ClientError:
            return False


class PromptOptimizationMixin:
    """
    Mixin class that can be added to existing AI model classes
    to provide automatic prompt optimization capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_client = PerfectPromptClient()
        self.auto_optimize = kwargs.get('auto_optimize_prompts', False)
        self.optimization_strategy = kwargs.get('optimization_strategy', 'comprehensive')
    
    def optimize_prompt_if_enabled(self, prompt: str) -> str:
        """
        Optimize prompt if auto-optimization is enabled.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Optimized prompt or original if optimization disabled/failed
        """
        if not self.auto_optimize:
            return prompt
        
        try:
            response = self.prompt_client.optimize_prompt(
                prompt, 
                strategy=self.optimization_strategy
            )
            
            if response.success and response.optimized_prompt:
                logger.info(f"Prompt optimized: {len(prompt)} -> {len(response.optimized_prompt)} chars")
                return response.optimized_prompt
            else:
                logger.warning("Prompt optimization failed, using original")
                return prompt
                
        except Exception as e:
            logger.error(f"Error in prompt optimization: {e}")
            return prompt
    
    def analyze_prompt_effectiveness(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze the effectiveness of a prompt.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            response = self.prompt_client.analyze_prompt(prompt)
            
            if response.success:
                return {
                    "metrics": response.metrics,
                    "analysis": response.analysis,
                    "suggestions": response.suggestions
                }
            else:
                return {"error": response.error_message}
                
        except Exception as e:
            return {"error": str(e)}


# Utility functions for easy integration

def quick_optimize(prompt: str, strategy: str = "comprehensive") -> str:
    """
    Quick function to optimize a prompt without creating a client instance.
    
    Args:
        prompt: The prompt to optimize
        strategy: Optimization strategy
        
    Returns:
        Optimized prompt or original if optimization fails
    """
    client = PerfectPromptClient()
    try:
        response = client.optimize_prompt(prompt, strategy)
        return response.optimized_prompt if response.success else prompt
    finally:
        client.close()


def quick_analyze(prompt: str) -> Dict[str, Any]:
    """
    Quick function to analyze a prompt without creating a client instance.
    
    Args:
        prompt: The prompt to analyze
        
    Returns:
        Analysis results dictionary
    """
    client = PerfectPromptClient()
    try:
        response = client.analyze_prompt(prompt)
        if response.success:
            return {
                "metrics": response.metrics,
                "analysis": response.analysis,
                "suggestions": response.suggestions
            }
        else:
            return {"error": response.error_message}
    finally:
        client.close()


async def async_quick_optimize(prompt: str, strategy: str = "comprehensive") -> str:
    """
    Async quick function to optimize a prompt.
    
    Args:
        prompt: The prompt to optimize
        strategy: Optimization strategy
        
    Returns:
        Optimized prompt or original if optimization fails
    """
    client = AsyncPerfectPromptClient()
    response = await client.optimize_prompt(prompt, strategy)
    return response.optimized_prompt if response.success else prompt
