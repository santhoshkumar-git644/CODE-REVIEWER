from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
import uuid
import logging

from ..schemas import ReportResponse, PaginatedResponse, CodeMetrics, ComplexityEstimate, SecurityIssue
from ..db import get_db
from ..models import AnalysisResult

router = APIRouter()
logger = logging.getLogger(__name__)

def db_model_to_response(db_obj: AnalysisResult) -> ReportResponse:
    """Helper to convert DB model to Pydantic schema."""
    
    # Parse JSON back to objects
    complexity = [ComplexityEstimate(**c) for c in db_obj.complexity_json] if isinstance(db_obj.complexity_json, list) else []
    security = [SecurityIssue(**s) for s in db_obj.security_issues_json] if isinstance(db_obj.security_issues_json, list) else []
    metrics = CodeMetrics(**db_obj.metrics_json) if isinstance(db_obj.metrics_json, dict) else CodeMetrics(function_count=0, max_nesting_depth=0, total_lines=0, avg_function_length=0, cyclomatic_complexity=0)
    
    return ReportResponse(
        analysis_id=db_obj.id,
        code_hash=db_obj.code_hash,
        language=db_obj.language,
        bug_probability=db_obj.bug_probability,
        risk_level=db_obj.risk_level,
        complexity_estimates=complexity,
        security_issues=security,
        metrics=metrics,
        analyzed_at=db_obj.created_at
    )

@router.get("/{analysis_id}", response_model=ReportResponse)
async def get_report(analysis_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Fetches a stored analysis report by ID."""
    
    result = await db.execute(select(AnalysisResult).where(AnalysisResult.id == analysis_id))
    db_obj = result.scalar_one_or_none()
    
    if not db_obj:
        raise HTTPException(status_code=404, detail="Analysis report not found.")
        
    return db_model_to_response(db_obj)

@router.get("/", response_model=PaginatedResponse)
async def list_reports(
    skip: int = Query(0, ge=0), 
    limit: int = Query(20, ge=1, le=100),
    language: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Lists all stored reports with pagination."""
    
    # Base query
    query = select(AnalysisResult)
    
    # Apply filters
    if language:
        query = query.where(AnalysisResult.language == language)
        
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()
    
    # Get items
    query = query.order_by(desc(AnalysisResult.created_at)).offset(skip).limit(limit)
    items_result = await db.execute(query)
    items = items_result.scalars().all()
    
    response_items = [db_model_to_response(item) for item in items]
    
    return PaginatedResponse(
        items=response_items,
        total=total,
        limit=limit,
        offset=skip
    )

@router.get("/stats/summary")
async def get_summary_stats(db: AsyncSession = Depends(get_db)):
    """Returns aggregated statistics across all reports."""
    
    try:
        # Total analyses
        total_result = await db.execute(select(func.count(AnalysisResult.id)))
        total = total_result.scalar_one()
        
        # Risk levels breakdown
        risk_query = select(AnalysisResult.risk_level, func.count(AnalysisResult.id)).group_by(AnalysisResult.risk_level)
        risk_result = await db.execute(risk_query)
        risk_breakdown = dict(risk_result.all())
        
        # Average bug probability
        avg_prob_result = await db.execute(select(func.avg(AnalysisResult.bug_probability)))
        avg_prob = avg_prob_result.scalar_one() or 0.0
        
        return {
            "total_analyses": total,
            "risk_levels": risk_breakdown,
            "average_bug_probability": float(avg_prob)
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")

@router.delete("/{analysis_id}")
async def delete_report(analysis_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Deletes a stored analysis report."""
    
    result = await db.execute(select(AnalysisResult).where(AnalysisResult.id == analysis_id))
    db_obj = result.scalar_one_or_none()
    
    if not db_obj:
        raise HTTPException(status_code=404, detail="Analysis report not found.")
        
    await db.delete(db_obj)
    await db.commit()
    
    return {"message": "Report deleted successfully", "id": str(analysis_id)}
