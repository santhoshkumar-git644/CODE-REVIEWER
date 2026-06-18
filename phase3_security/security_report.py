import json
from dataclasses import dataclass
from typing import List, Dict
from .pattern_matcher import Finding

@dataclass
class SecuritySummary:
    total_findings: int
    high_severity: int
    medium_severity: int
    low_severity: int
    risk_score: float
    risk_level: str

    def to_dict(self):
        return {
            "total_findings": self.total_findings,
            "high_severity": self.high_severity,
            "medium_severity": self.medium_severity,
            "low_severity": self.low_severity,
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level
        }

def calculate_risk_score(findings: List[Finding]) -> float:
    """Calculates a risk score from 0 to 10 based on findings."""
    if not findings:
        return 0.0
        
    score = 0.0
    for f in findings:
        if f.severity == 'HIGH':
            score += 3.0
        elif f.severity == 'MEDIUM':
            score += 1.5
        elif f.severity == 'LOW':
            score += 0.5
            
    # Cap at 10.0
    return min(10.0, score)

def get_risk_level(score: float) -> str:
    if score >= 7.0:
        return "CRITICAL"
    elif score >= 4.0:
        return "HIGH"
    elif score >= 1.5:
        return "MEDIUM"
    elif score > 0:
        return "LOW"
    return "SECURE"

def generate_summary(findings: List[Finding]) -> SecuritySummary:
    """Generates a summary of all security findings."""
    high = sum(1 for f in findings if f.severity == 'HIGH')
    medium = sum(1 for f in findings if f.severity == 'MEDIUM')
    low = sum(1 for f in findings if f.severity == 'LOW')
    
    score = calculate_risk_score(findings)
    level = get_risk_level(score)
    
    return SecuritySummary(
        total_findings=len(findings),
        high_severity=high,
        medium_severity=medium,
        low_severity=low,
        risk_score=score,
        risk_level=level
    )

def generate_json_report(findings: List[Finding]) -> str:
    """Generates a structured JSON report."""
    summary = generate_summary(findings)
    
    report = {
        "summary": summary.to_dict(),
        "findings": [f.to_dict() for f in findings]
    }
    
    return json.dumps(report, indent=2)

def generate_text_report(findings: List[Finding]) -> str:
    """Generates a human-readable text report."""
    summary = generate_summary(findings)
    
    lines = []
    lines.append("=" * 60)
    lines.append(" SECURITY VULNERABILITY REPORT ")
    lines.append("=" * 60)
    lines.append(f"Risk Level: {summary.risk_level} (Score: {summary.risk_score:.1f}/10.0)")
    lines.append(f"Total Findings: {summary.total_findings} (High: {summary.high_severity}, Med: {summary.medium_severity}, Low: {summary.low_severity})")
    lines.append("-" * 60)
    
    if not findings:
        lines.append(" No vulnerabilities detected. Great job!")
        return "\n".join(lines)
        
    for i, f in enumerate(findings, 1):
        lines.append(f"[{i}] [{f.severity}] {f.rule_name} ({f.cwe_id})")
        lines.append(f"    Location: {f.file_path}:{f.line_number}")
        if f.code_snippet:
            lines.append(f"    Code:     {f.code_snippet}")
        lines.append(f"    Details:  {f.description}")
        lines.append(f"    Fix:      {f.remediation}")
        lines.append("-" * 60)
        
    return "\n".join(lines)
