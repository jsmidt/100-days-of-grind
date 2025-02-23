.PHONY: install test lint coverage docs build release

install:
	uv venv
	uv pip install -r requirements.txt pytest pytest-cov ruff mkdocs mkdocs-material build twine bump2version

test:
	pytest tests/

lint:
	ruff check src tests

format:
	ruff format src tests

coverage:
	pytest --cov=hello_uv --cov-report=term

docs:
	mkdocs serve

build:
	python -m build

release:
	bump2version patch
	git push --follow-tags
