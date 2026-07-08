.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	python main.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/supersense.db
