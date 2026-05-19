import ast
from dataclasses import dataclass
from typing import List, Optional
from .vuln_rules import VulnerabilityRule, get_rules_by_language

@dataclass
class Finding:
    rule_id: str
    rule_name: str
    severity: str
    cwe_id: str
    file_path: str
    line_number: int
    column: int
    code_snippet: str
    description: str
    remediation: str

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "cwe_id": self.cwe_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "description": self.description,
            "remediation": self.remediation
        }

def match_regex_patterns(code: str, rules: List[VulnerabilityRule], file_path: str = "unknown") -> List[Finding]:
    findings = []
    lines = code.split('\n')
    
    for rule in rules:
        if rule.pattern_type != 'regex' or not rule.regex_obj:
            continue
            
        for i, line in enumerate(lines):
            match = rule.regex_obj.search(line)
            if match:
                findings.append(Finding(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    cwe_id=rule.cwe_id,
                    file_path=file_path,
                    line_number=i + 1,
                    column=match.start(),
                    code_snippet=line.strip()[:100],  # Truncate snippet
                    description=rule.description,
                    remediation=rule.remediation
                ))
    return findings

class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self, rules: List[VulnerabilityRule], file_path: str, code_lines: List[str]):
        self.rules = rules
        self.file_path = file_path
        self.code_lines = code_lines
        self.findings = []
        
        # Pre-filter AST rules
        self.ast_rules = {r.pattern: r for r in rules if r.pattern_type == 'ast'}

    def visit_Call(self, node):
        # Match 'eval_call' pattern
        if 'eval_call' in self.ast_rules:
            if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                self._add_finding(self.ast_rules['eval_call'], node)
                
        # Match 'os_system_call' pattern
        if 'os_system_call' in self.ast_rules:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'os' and node.func.attr == 'system':
                    self._add_finding(self.ast_rules['os_system_call'], node)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'subprocess' and node.func.attr == 'Popen':
                    for kw in node.keywords:
                        if kw.arg == 'shell' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            self._add_finding(self.ast_rules['os_system_call'], node)

        self.generic_visit(node)

    def _add_finding(self, rule: VulnerabilityRule, node: ast.AST):
        line_num = getattr(node, 'lineno', 1)
        col_offset = getattr(node, 'col_offset', 0)
        snippet = self.code_lines[line_num - 1].strip()[:100] if 0 < line_num <= len(self.code_lines) else ""
        
        self.findings.append(Finding(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            cwe_id=rule.cwe_id,
            file_path=self.file_path,
            line_number=line_num,
            column=col_offset,
            code_snippet=snippet,
            description=rule.description,
            remediation=rule.remediation
        ))

def match_ast_patterns(code: str, rules: List[VulnerabilityRule], file_path: str = "unknown") -> List[Finding]:
    """Matches AST-based security patterns for Python code."""
    try:
        tree = ast.parse(code)
        code_lines = code.split('\n')
        visitor = SecurityASTVisitor(rules, file_path, code_lines)
        visitor.visit(tree)
        return visitor.findings
    except SyntaxError:
        # Cannot parse invalid code
        return []

def scan_code(code: str, language: str = 'python', file_path: str = "unknown") -> List[Finding]:
    """Main scanning function that runs both regex and AST checks."""
    rules = get_rules_by_language(language)
    findings = []
    
    # 1. Regex checks
    findings.extend(match_regex_patterns(code, rules, file_path))
    
    # 2. AST checks (Python only for now using built-in ast)
    if language.lower() == 'python':
        findings.extend(match_ast_patterns(code, rules, file_path))
        
    return findings
