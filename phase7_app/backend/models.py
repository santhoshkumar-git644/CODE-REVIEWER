from sqlalchemy import Column, String, Float, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
import datetime

Base = declarative_base()

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code_hash = Column(String(64), index=True, nullable=False)
    language = Column(String(20), nullable=False)
    code_snippet = Column(Text, nullable=False)
    bug_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    
    # Store complex structures as JSON
    complexity_json = Column(JSON, nullable=False)
    security_issues_json = Column(JSON, nullable=False)
    metrics_json = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
