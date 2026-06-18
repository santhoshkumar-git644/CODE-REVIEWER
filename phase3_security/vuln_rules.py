import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Pattern

@dataclass
class VulnerabilityRule:
    rule_id: str
    name: str
    description: str
    severity: str  # HIGH, MEDIUM, LOW
    cwe_id: str
    pattern_type: str  # 'regex' or 'ast'
    pattern: str
    language: str
    remediation: str
    regex_obj: Optional[Pattern] = None

    def __post_init__(self):
        if self.pattern_type == 'regex' and self.pattern:
            self.regex_obj = re.compile(self.pattern)

# Define 15+ rules covering the requested CWEs
RULES = [
    VulnerabilityRule(
        rule_id="SEC-001",
        name="SQL Injection",
        description="Directly concatenating strings into SQL queries can lead to SQL injection.",
        severity="HIGH",
        cwe_id="CWE-89",
        pattern_type="regex",
        pattern=r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC)\s+.*?\+.*",
        language="all",
        remediation="Use parameterized queries or prepared statements instead of string concatenation."
    ),
    VulnerabilityRule(
        rule_id="SEC-002",
        name="Command Injection",
        description="Executing OS commands with untrusted input can lead to remote code execution.",
        severity="HIGH",
        cwe_id="CWE-78",
        pattern_type="ast",
        pattern="os_system_call", # Abstract representation for AST matcher
        language="python",
        remediation="Avoid using os.system or subprocess with shell=True. Use subprocess.run with a list of arguments."
    ),
    VulnerabilityRule(
        rule_id="SEC-003",
        name="Path Traversal",
        description="Improper limitation of a pathname to a restricted directory.",
        severity="HIGH",
        cwe_id="CWE-22",
        pattern_type="regex",
        pattern=r"\.\.[\\/]",
        language="all",
        remediation="Sanitize file paths and validate against a whitelist of allowed paths."
    ),
    VulnerabilityRule(
        rule_id="SEC-004",
        name="Buffer Overflow (strcpy)",
        description="Use of unsafe functions like strcpy can lead to buffer overflows.",
        severity="HIGH",
        cwe_id="CWE-120",
        pattern_type="regex",
        pattern=r"\bstrcpy\s*\(",
        language="c",
        remediation="Use safer alternatives like strncpy or strlcpy with proper bounds checking."
    ),
    VulnerabilityRule(
        rule_id="SEC-005",
        name="Use After Free",
        description="Referencing memory after it has been freed can cause crashes or code execution.",
        severity="HIGH",
        cwe_id="CWE-416",
        pattern_type="regex",
        pattern=r"\bfree\s*\([^)]+\).*?\b\w+\s*->\s*\w+",
        language="c",
        remediation="Set pointers to NULL immediately after freeing them."
    ),
    VulnerabilityRule(
        rule_id="SEC-006",
        name="Null Pointer Dereference",
        description="Dereferencing a pointer that might be null.",
        severity="MEDIUM",
        cwe_id="CWE-476",
        pattern_type="regex",
        pattern=r"\b\w+\s*=\s*NULL;.*?\*\w+",
        language="c",
        remediation="Check pointers for NULL before dereferencing them."
    ),
    VulnerabilityRule(
        rule_id="SEC-007",
        name="Hardcoded Credentials",
        description="Hardcoding passwords or keys in source code.",
        severity="HIGH",
        cwe_id="CWE-798",
        pattern_type="regex",
        pattern=r"(?i)(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
        language="all",
        remediation="Store credentials in environment variables or a secure vault."
    ),
    VulnerabilityRule(
        rule_id="SEC-008",
        name="Insecure Randomness",
        description="Using weak random number generators for security-sensitive operations.",
        severity="MEDIUM",
        cwe_id="CWE-330",
        pattern_type="regex",
        pattern=r"\b(random\(\)|rand\(\)|Math\.random\(\))",
        language="all",
        remediation="Use a cryptographically secure pseudo-random number generator (CSPRNG)."
    ),
    VulnerabilityRule(
        rule_id="SEC-009",
        name="Missing Input Validation",
        description="Missing validation on user input.",
        severity="MEDIUM",
        cwe_id="CWE-20",
        pattern_type="regex",
        pattern=r"\brequest\.(GET|POST)\[['\"].*?['\"]\]",
        language="python",
        remediation="Always validate and sanitize user input before processing it."
    ),
    VulnerabilityRule(
        rule_id="SEC-010",
        name="Cross-Site Scripting (XSS)",
        description="Improper neutralization of input during web page generation.",
        severity="HIGH",
        cwe_id="CWE-79",
        pattern_type="regex",
        pattern=r"innerHTML\s*=",
        language="javascript",
        remediation="Use textContent or escape HTML entities before rendering user input."
    ),
    VulnerabilityRule(
        rule_id="SEC-011",
        name="Insecure Deserialization",
        description="Deserializing untrusted data can lead to remote code execution.",
        severity="HIGH",
        cwe_id="CWE-502",
        pattern_type="regex",
        pattern=r"\b(pickle\.loads|yaml\.load)\s*\(",
        language="python",
        remediation="Avoid deserializing untrusted data. Use safer formats like JSON or yaml.safe_load."
    ),
    VulnerabilityRule(
        rule_id="SEC-012",
        name="Uncontrolled Resource Consumption",
        description="Failure to properly limit resource usage, leading to DoS.",
        severity="MEDIUM",
        cwe_id="CWE-400",
        pattern_type="regex",
        pattern=r"\bwhile\s*\(\s*True\s*\)",
        language="all",
        remediation="Ensure loops have clear termination conditions or timeout mechanisms."
    ),
    VulnerabilityRule(
        rule_id="SEC-013",
        name="Information Exposure",
        description="Exposing sensitive information through error messages.",
        severity="LOW",
        cwe_id="CWE-200",
        pattern_type="regex",
        pattern=r"\bprint\s*\(\s*Exception\s*\)|\bconsole\.error\s*\(.*err.*\)",
        language="all",
        remediation="Log sensitive errors securely and return generic error messages to users."
    ),
    VulnerabilityRule(
        rule_id="SEC-014",
        name="Improper Error Handling",
        description="Catching generic exceptions can mask underlying security issues.",
        severity="LOW",
        cwe_id="CWE-755",
        pattern_type="regex",
        pattern=r"\bexcept\s+Exception\s*:",
        language="python",
        remediation="Catch specific exceptions rather than a broad Exception class."
    ),
    VulnerabilityRule(
        rule_id="SEC-015",
        name="Use of Deprecated Functions",
        description="Using functions that have been deprecated or declared obsolete.",
        severity="LOW",
        cwe_id="CWE-477",
        pattern_type="regex",
        pattern=r"\b(gets|tempnam|mktemp)\s*\(",
        language="c",
        remediation="Use modern, secure alternatives (e.g., fgets instead of gets)."
    ),
    VulnerabilityRule(
        rule_id="SEC-016",
        name="Eval Usage",
        description="Using eval() can execute arbitrary code.",
        severity="HIGH",
        cwe_id="CWE-94",
        pattern_type="ast",
        pattern="eval_call",
        language="python",
        remediation="Avoid eval(). Use ast.literal_eval() for parsing data."
    )
]

def get_rules_by_severity(severity: str) -> List[VulnerabilityRule]:
    """Returns a list of rules matching the given severity."""
    return [rule for rule in RULES if rule.severity.upper() == severity.upper()]

def get_rules_by_language(language: str) -> List[VulnerabilityRule]:
    """Returns rules applicable to a specific language (including 'all')."""
    return [rule for rule in RULES if rule.language == 'all' or rule.language.lower() == language.lower()]

def get_rule_by_cwe(cwe_id: str) -> Optional[VulnerabilityRule]:
    """Returns the rule matching the given CWE ID."""
    for rule in RULES:
        if rule.cwe_id.upper() == cwe_id.upper():
            return rule
    return None

def get_all_rules() -> List[VulnerabilityRule]:
    """Returns all defined rules."""
    return RULES
