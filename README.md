# AI Code Review & Bug Risk Analyzer

An intelligent static analysis and bug prediction tool using AST parsing, Machine Learning (Random Forest / XGBoost), and Deep Learning (CodeBERT).

## Features

- **Static Analysis**: Computes metrics like Cyclomatic Complexity, Max Nesting Depth, and Function counts.
- **Time Complexity Estimation**: Detects loops and recursive patterns to estimate Big-O complexity.
- **Security Vulnerability Detection**: Identifies 15+ common security flaws based on CWE standards (e.g., SQL Injection, Command Injection, XSS).
- **Bug Prediction Model**: Uses ML models and fine-tuned CodeBERT to predict the probability of a function containing a bug.
- **FastAPI Backend**: High-performance async API with PostgreSQL database support.
- **Interactive UI**: Beautiful dark-mode dashboard for submitting code and viewing analysis results.

## Quick Start (Docker)

1. Clone the repository
2. Run `docker-compose up --build`
3. Open `http://localhost:8000` (assuming frontend is served or open `phase7_app/frontend/index.html` directly in your browser)
4. API docs available at `http://localhost:8000/docs`

## Local Development

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Copy environment variables: `cp .env.example .env`
5. Run the API: `uvicorn phase7_app.backend.main:app --reload`
6. Open `phase7_app/frontend/index.html` in your browser.

## Project Structure

- `phase1_static_analyzer/`: AST parsing and basic metric calculation
- `phase2_complexity/`: Time complexity estimation
- `phase3_security/`: Rule-based vulnerability detection
- `phase4_dataset/`: Dataset collection and feature extraction
- `phase5_ml_model/`: ML models (RF, XGBoost) and hyperparameter tuning
- `phase6_codebert/`: CodeBERT embeddings and deep learning fine-tuning
- `phase7_app/`: FastAPI backend and vanilla JS/HTML frontend

## License

MIT License
