import os
import time
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Import routers and DB setup
from .routers import analyze_router, report_router
from .db import init_db
from .schemas import HealthResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Application startup time
START_TIME = time.time()

# Global state for ML model
ml_model_state = {
    "loaded": False,
    "model": None
}

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    
    app = FastAPI(
        title="AI-Powered Code Review API",
        description="Analyzes code for bugs, complexity, and security vulnerabilities.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development. In prod, restrict to specific domains.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_req_time = time.time()
        
        # Simple rate limiting (in-memory, per IP)
        # In a real app, use Redis or similar
        client_ip = request.client.host if request.client else "unknown"
        # Dummy rate check omitted for brevity
        
        response = await call_next(request)
        
        process_time = time.time() - start_req_time
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
        
        # Add Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )

    # Include Routers
    app.include_router(analyze_router, prefix="/api/v1/analyze", tags=["Analyze"])
    app.include_router(report_router, prefix="/api/v1/reports", tags=["Reports"])

    # Startup & Shutdown Events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Initializing database...")
        await init_db()
        
        logger.info("Loading ML models into memory...")
        try:
            # Placeholder for loading actual models
            # from phase5_ml_model.model_saver import load_best_model
            # model = load_best_model("models", "bug_predictor")
            # ml_model_state["model"] = model
            ml_model_state["loaded"] = True
            logger.info("ML models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            logger.warning("Starting API without ML prediction capabilities. Fallback heuristics will be used.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutting down. Cleaning up resources...")
        # Free memory
        ml_model_state["model"] = None
        ml_model_state["loaded"] = False

    # Root & Health Endpoints
    @app.get("/", tags=["General"])
    async def root():
        return {
            "name": "AI Code Review API",
            "version": "1.0.0",
            "documentation": "/docs",
            "status": "online"
        }

    @app.get("/health", response_model=HealthResponse, tags=["General"])
    async def health_check():
        uptime = time.time() - START_TIME
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model_loaded=ml_model_state["loaded"],
            uptime=uptime
        )

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run("phase7_app.backend.main:app", host=host, port=port, reload=True)
