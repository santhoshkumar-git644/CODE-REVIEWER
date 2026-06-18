from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="The source code to analyze")
    language: str = Field("python", description="Programming language (e.g., python, c, javascript)")

class SecurityIssue(BaseModel):
    rule_name: str
    severity: str
    cwe_id: str
    line_number: int
    description: str
    remediation: str

class ComplexityEstimate(BaseModel):
    function_name: str
    complexity_class: str
    confidence: float

class CodeMetrics(BaseModel):
    function_count: int
    max_nesting_depth: int
    total_lines: int
    avg_function_length: float
    cyclomatic_complexity: int

class AnalyzeResponse(BaseModel):
    analysis_id: uuid.UUID
    bug_probability: float
    risk_level: str
    complexity_estimates: List[ComplexityEstimate]
    security_issues: List[SecurityIssue]
    metrics: CodeMetrics
    analyzed_at: datetime

class ReportResponse(AnalyzeResponse):
    code_hash: str
    language: str

class PaginatedResponse(BaseModel):
    items: List[ReportResponse]
    total: int
    limit: int
    offset: int

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    uptime: float
