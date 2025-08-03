"""
FastAPI server for Perfect Prompt optimization API.
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from loguru import logger
import os
from contextlib import asynccontextmanager

from ..core.prompt_analyzer import PromptAnalyzer
from ..core.prompt_optimizer import PromptOptimizer
from ..models.prompt_model import PromptModel


# Pydantic models for API
class PromptAnalyzeRequest(BaseModel):
    prompt: str
    include_suggestions: bool = True


class PromptOptimizeRequest(BaseModel):
    prompt: str
    strategy: str = "comprehensive"
    target_metrics: Optional[Dict[str, float]] = None
    include_analysis: bool = True


class BatchOptimizeRequest(BaseModel):
    prompts: List[str]
    strategy: str = "comprehensive"


class ComparePromptsRequest(BaseModel):
    prompts: List[str]


class PromptAnalyzeResponse(BaseModel):
    success: bool
    prompt: str
    analysis: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


class PromptOptimizeResponse(BaseModel):
    success: bool
    original_prompt: str
    optimized_prompt: Optional[str] = None
    improvement_score: Optional[float] = None
    optimization_steps: Optional[List[str]] = None
    analysis: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


class BatchOptimizeResponse(BaseModel):
    success: bool
    results: List[PromptOptimizeResponse]
    total_processed: int


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


class APIInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: List[str]
    strategies: List[str]
    features: List[str]
    privacy_policy: str


# Global components
analyzer = None
optimizer = None
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    global analyzer, optimizer
    
    logger.info("Initializing Perfect Prompt API components...")
    try:
        analyzer = PromptAnalyzer()
        optimizer = PromptOptimizer(analyzer)
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Perfect Prompt API")


# Create FastAPI app
app = FastAPI(
    title="Perfect Prompt API",
    description="AI-powered prompt optimization system for embedding in other AI models",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key if authentication is enabled."""
    if not os.getenv("REQUIRE_API_KEY", "false").lower() == "true":
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    valid_api_key = os.getenv("API_KEY")
    if not valid_api_key or credentials.credentials != valid_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


def convert_metrics_to_dict(metrics) -> Dict[str, float]:
    """Convert metrics object to dictionary."""
    return {
        "length": metrics.length,
        "complexity_score": metrics.complexity_score,
        "clarity_score": metrics.clarity_score,
        "specificity_score": metrics.specificity_score,
        "sentiment_score": metrics.sentiment_score,
        "readability_score": metrics.readability_score,
        "semantic_density": metrics.semantic_density,
        "instruction_clarity": metrics.instruction_clarity,
        "context_richness": metrics.context_richness,
        "efficiency_score": metrics.efficiency_score
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components={
            "analyzer": "ready" if analyzer else "not_ready",
            "optimizer": "ready" if optimizer else "not_ready"
        }
    )


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    """Serve the privacy policy."""
    try:
        import os
        # Get the path to the privacy policy file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        privacy_file_path = os.path.join(current_dir, "..", "..", "deploy", "privacy-policy.html")
        
        with open(privacy_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Privacy Policy</h1><p>Privacy policy not found. Please contact hant2953@gmail.com for information.</p>",
            status_code=404
        )


@app.get("/api/v1/info", response_model=APIInfoResponse)
async def get_api_info():
    """Get API information and capabilities."""
    return APIInfoResponse(
        name="Perfect Prompt API",
        version="0.1.0",
        description="AI-powered prompt optimization system",
        endpoints=[
            "/api/v1/analyze",
            "/api/v1/optimize",
            "/api/v1/batch-optimize",
            "/api/v1/compare",
            "/health",
            "/privacy",
            "/api/v1/info"
        ],
        strategies=[
            "clarity_focused",
            "efficiency_focused", 
            "comprehensive",
            "creative"
        ],
        features=[
            "Prompt Analysis",
            "Prompt Optimization",
            "Batch Processing",
            "Prompt Comparison",
            "Multiple Strategies",
            "Target Metrics",
            "API Integration"
        ],
        privacy_policy="https://perfect-prompt-api-production.up.railway.app/privacy"
    )


@app.post("/api/v1/analyze", response_model=PromptAnalyzeResponse)
async def analyze_prompt(
    request: PromptAnalyzeRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Analyze a prompt for effectiveness metrics."""
    try:
        logger.info(f"Analyzing prompt: {request.prompt[:50]}...")
        
        analysis = analyzer.analyze_prompt(request.prompt)
        
        # Convert analysis to dictionary format
        analysis_dict = {
            "identified_patterns": analysis.identified_patterns,
            "potential_issues": analysis.potential_issues,
            "optimization_opportunities": analysis.optimization_opportunities
        }
        
        metrics_dict = convert_metrics_to_dict(analysis.metrics)
        
        return PromptAnalyzeResponse(
            success=True,
            prompt=request.prompt,
            analysis=analysis_dict,
            suggestions=analysis.suggestions if request.include_suggestions else None,
            metrics=metrics_dict
        )
        
    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        return PromptAnalyzeResponse(
            success=False,
            prompt=request.prompt,
            error_message=str(e)
        )


@app.post("/api/v1/optimize", response_model=PromptOptimizeResponse)
async def optimize_prompt(
    request: PromptOptimizeRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Optimize a prompt for better effectiveness."""
    try:
        logger.info(f"Optimizing prompt with strategy: {request.strategy}")
        
        if request.target_metrics:
            # Use target-based optimization
            result = optimizer.optimize_for_target(
                request.prompt,
                request.target_metrics
            )
        else:
            # Use strategy-based optimization
            result = optimizer.optimize_prompt(
                request.prompt,
                request.strategy
            )
        
        response_data = {
            "success": True,
            "original_prompt": result.original_prompt,
            "optimized_prompt": result.optimized_prompt,
            "improvement_score": result.improvement_score,
            "optimization_steps": result.optimization_steps
        }
        
        if request.include_analysis:
            analysis_dict = {
                "identified_patterns": result.final_analysis.identified_patterns,
                "potential_issues": result.final_analysis.potential_issues,
                "optimization_opportunities": result.final_analysis.optimization_opportunities
            }
            response_data["analysis"] = analysis_dict
            response_data["metrics"] = convert_metrics_to_dict(result.final_analysis.metrics)
        
        return PromptOptimizeResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        return PromptOptimizeResponse(
            success=False,
            original_prompt=request.prompt,
            error_message=str(e)
        )


@app.post("/api/v1/batch-optimize", response_model=BatchOptimizeResponse)
async def batch_optimize_prompts(
    request: BatchOptimizeRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Optimize multiple prompts in batch."""
    try:
        logger.info(f"Batch optimizing {len(request.prompts)} prompts")
        
        results = optimizer.batch_optimize(request.prompts, request.strategy)
        
        response_results = []
        for result in results:
            response_results.append(PromptOptimizeResponse(
                success=True,
                original_prompt=result.original_prompt,
                optimized_prompt=result.optimized_prompt,
                improvement_score=result.improvement_score,
                optimization_steps=result.optimization_steps,
                analysis={
                    "identified_patterns": result.final_analysis.identified_patterns,
                    "potential_issues": result.final_analysis.potential_issues,
                    "optimization_opportunities": result.final_analysis.optimization_opportunities
                },
                metrics=convert_metrics_to_dict(result.final_analysis.metrics)
            ))
        
        return BatchOptimizeResponse(
            success=True,
            results=response_results,
            total_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in batch optimization: {e}")
        return BatchOptimizeResponse(
            success=False,
            results=[],
            total_processed=0
        )


@app.post("/api/v1/compare")
async def compare_prompts(
    request: ComparePromptsRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Compare multiple prompts and provide recommendations."""
    try:
        logger.info(f"Comparing {len(request.prompts)} prompts")
        
        comparison_result = analyzer.compare_prompts(request.prompts)
        
        # Convert comparison result to JSON-serializable format
        result = {
            "success": True,
            "total_prompts": len(request.prompts),
            "best_overall": {
                "index": comparison_result["best_overall"]["index"],
                "prompt": comparison_result["best_overall"]["prompt"],
                "score": comparison_result["best_overall"]["score"]
            },
            "best_metrics": {},
            "recommendations": comparison_result["recommendations"],
            "detailed_analyses": []
        }
        
        # Convert best metrics
        for metric_name, metric_data in comparison_result["best_metrics"].items():
            result["best_metrics"][metric_name] = {
                "index": metric_data["index"],
                "prompt": metric_data["prompt"],
                "score": metric_data["score"]
            }
        
        # Convert detailed analyses
        for analysis in comparison_result["analyses"]:
            result["detailed_analyses"].append({
                "prompt": analysis.prompt,
                "metrics": convert_metrics_to_dict(analysis.metrics),
                "suggestions": analysis.suggestions,
                "identified_patterns": analysis.identified_patterns,
                "potential_issues": analysis.potential_issues
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error comparing prompts: {e}")
        return {
            "success": False,
            "error_message": str(e)
        }


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/perfect_prompt_api.log", rotation="1 day", retention="7 days")
    
    # Run the server
    uvicorn.run(
        "perfect_prompt.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
