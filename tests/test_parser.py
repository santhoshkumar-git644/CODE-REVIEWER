import pytest
from phase1_static_analyzer.parser import parse_code
from phase1_static_analyzer.metrics import compute_all_metrics

def test_python_parsing(sample_python_code):
    tree = parse_code(sample_python_code, "python")
    assert tree is not None
    
def test_metrics_computation(sample_python_code):
    metrics = compute_all_metrics(sample_python_code, "python")
    assert metrics.function_count == 2
    assert metrics.max_nesting_depth >= 2 # for loops
    assert metrics.total_lines > 5
