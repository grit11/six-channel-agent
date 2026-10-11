# 兼容 Windows PowerShell，后端命令固定使用已存在的 langgraph 环境。
SHELL := pwsh

.PHONY: dev test lint format-check typecheck ci index

index:
	conda run -n langgraph python scripts/index_knowledge.py

dev:
	conda run -n langgraph python scripts/start_console.py

test:
	conda run -n langgraph python -m pytest backend/tests

lint:
	conda run -n langgraph python -m ruff check backend scripts eval

format-check:
	conda run -n langgraph python -m black --config backend/pyproject.toml --check backend scripts eval

typecheck:
	conda run -n langgraph python -m mypy --config-file backend/pyproject.toml backend/app

ci:
	pwsh -File scripts/ci.ps1
