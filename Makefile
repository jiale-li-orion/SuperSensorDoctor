.PHONY: install install-e2e run test clean lock sync

install: ## create .venv + install locked deps (runtime + dev)
	uv sync

install-e2e: ## + browser E2E group
	uv sync --group e2e
	uv run playwright install chromium

sync:
	uv sync

lock:
	uv lock

run:
	uv run python main.py

test:
	uv run pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/supersense.db
