import pytest
from phase3_security.pattern_matcher import scan_code

def test_vulnerability_detection(vulnerable_python_code):
    findings = scan_code(vulnerable_python_code, "python")
    
    # Should detect eval and os.system
    assert len(findings) >= 2
    
    rule_names = [f.rule_name for f in findings]
    assert any("Command Injection" in name or "os.system" in name.lower() for name in rule_names)
    assert any("eval" in name.lower() for name in rule_names)
