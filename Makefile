# Makefile for Table Comparison Project

.PHONY: help install install-dev test test-unit test-integration clean lint format check-deps run-example setup

# Default target
help:
	@echo "Table Comparison Project - Available Commands:"
	@echo "=============================================="
	@echo "setup          - Complete project setup (install + check dependencies)"
	@echo "install        - Install project dependencies"
	@echo "install-dev    - Install development dependencies"
	@echo "test           - Run all tests"
	@echo "test-unit      - Run unit tests only"
	@echo "test-deps      - Check test dependencies"
	@echo "clean          - Clean up generated files"
	@echo "lint           - Run code linting"
	@echo "format         - Format code with black"
	@echo "check-deps     - Check system dependencies"
	@echo "run-example    - Run example comparison"
	@echo "notebook       - Start Jupyter notebook"
	@echo "git-ready      - Prepare repository for Git"

# Setup and installation
setup: install check-deps
	@echo "✓ Project setup complete!"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	pip install pytest pytest-cov black flake8 mypy jupyter notebook

# Testing
test:
	@echo "Running all tests..."
	python tests/run_tests.py

test-unit:
	@echo "Running unit tests..."
	python tests/run_tests.py --tests embeddings comparisons utils

test-deps:
	@echo "Checking test dependencies..."
	python tests/run_tests.py --check-deps

# Code quality
lint:
	@echo "Running linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503; \
	else \
		echo "flake8 not installed. Run 'make install-dev' first."; \
	fi

format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ tests/ main.py config.py; \
	else \
		echo "black not installed. Run 'make install-dev' first."; \
	fi

# System checks
check-deps:
	@echo "Checking system dependencies..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python tests/run_tests.py --check-deps

# Utility commands
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	rm -rf build/ dist/ .coverage htmlcov/

run-example:
	@echo "Running example comparison..."
	@if [ -f table_embeddings.ipynb ]; then \
		echo "Opening main notebook..."; \
		jupyter notebook table_embeddings.ipynb; \
	else \
		echo "Example data not found. Please run the notebook first."; \
	fi

notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook

# Git preparation
git-ready: clean format
	@echo "Preparing repository for Git..."
	@echo "✓ Cleaned temporary files"
	@echo "✓ Formatted code"
	@if command -v git >/dev/null 2>&1; then \
		if [ -d .git ]; then \
			echo "Git repository already initialized"; \
		else \
			echo "Initializing Git repository..."; \
			git init; \
			git add .; \
			echo "Files staged. Ready for initial commit."; \
			echo "Run: git commit -m 'Initial commit: Table comparison toolkit'"; \
		fi \
	else \
		echo "Git not found. Install Git to initialize repository."; \
	fi

# Development workflow
dev-setup: install-dev check-deps
	@echo "Development environment setup complete!"
	@echo "Available commands:"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Check code style"
	@echo "  make format    - Format code"
	@echo "  make notebook  - Start Jupyter"

# Quick commands
quick-test: test-deps test-unit

quick-check: lint check-deps

# Documentation
docs:
	@echo "Documentation files:"
	@echo "  README.md           - Main project documentation"
	@echo "  MERMAID_FLOWS.md    - Process flow diagrams"
	@echo "  requirements.txt    - Python dependencies"
	@echo "  config.py          - Configuration settings"

# Package building (if needed)
build:
	@echo "Building package..."
	python -m build

# Ollama setup helpers
ollama-setup:
	@echo "Setting up Ollama models..."
	@echo "Make sure Ollama is installed and running"
	@echo "Installing embedding models..."
	@if command -v ollama >/dev/null 2>&1; then \
		ollama pull mxbai-embed-large; \
		ollama pull nomic-embed-text; \
		echo "✓ Embedding models installed"; \
	else \
		echo "Ollama not found. Please install Ollama first."; \
		echo "Visit: https://ollama.ai"; \
	fi

ollama-check:
	@echo "Checking Ollama status..."
	@if command -v ollama >/dev/null 2>&1; then \
		curl -s http://localhost:11434/api/tags || echo "Ollama not running. Start with: ollama serve"; \
	else \
		echo "Ollama not installed."; \
	fi

# Complete setup for new users
first-time-setup: install ollama-setup check-deps
	@echo "================================================"
	@echo "First-time setup complete!"
	@echo "================================================"
	@echo "Next steps:"
	@echo "1. Start Jupyter: make notebook"
	@echo "2. Open table_embeddings.ipynb"
	@echo "3. Run the notebook cells"
	@echo "4. Explore the results and documentation"
	@echo "================================================"