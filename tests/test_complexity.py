import pytest
from phase1_static_analyzer.parser import parse_code
from phase2_complexity.loop_analyzer import estimate_all_functions

def test_complexity_estimation(sample_python_code):
    tree = parse_code(sample_python_code, "python")
    estimates = estimate_all_functions(tree.root_node if hasattr(tree, 'root_node') else tree)
    
    assert len(estimates) == 2
    
    # Check if complex_function is identified as O(n^2)
    complex_est = next((e for e in estimates if e.function_name == 'complex_function'), None)
    assert complex_est is not None
    assert 'n²' in complex_est.complexity_class or 'n^2' in complex_est.complexity_class
