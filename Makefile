.PHONY: help install test lint format clean preprocess train eval

help:
	@echo "Available commands:"
	@echo "  make install     - Install all dependencies"
	@echo "  make test        - Run tests with coverage"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code with black and isort"
	@echo "  make clean       - Remove generated files"
	@echo "  make preprocess  - Run data preprocessing"
	@echo "  make train       - Train the model"
	@echo "  make eval        - Evaluate the model"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/ train.py eval.py preprocess.py
	isort src/ tests/ train.py eval.py preprocess.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

preprocess:
	python preprocess.py

train:
	python train.py --data loops_035 --batch_size 64 --lr 0.001 --epochs 100

eval:
	python eval.py --data loops_035 --plot
