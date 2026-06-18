import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure the root directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase7_app.backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_python_code():
    return """
def hello_world():
    print("Hello, world!")
    
def complex_function(data):
    result = 0
    for i in range(10):
        for j in range(10):
            result += i * j
    return result
"""

@pytest.fixture
def vulnerable_python_code():
    return """
import os

def run_command(user_input):
    # Vulnerable to command injection
    os.system("echo " + user_input)
    
    # Vulnerable to eval
    data = eval(user_input)
    return data
"""
