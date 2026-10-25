.PHONY: install dev lint format test check-data classify denoise caption

install:
	pip install -r requirements.txt
	pip install -e .

dev:
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

check-data:
	tounum check-data

classify:
	tounum classify train

denoise:
	tounum denoise train

caption:
	tounum caption train
