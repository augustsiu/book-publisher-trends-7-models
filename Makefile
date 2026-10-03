.PHONY: install test lint smoke run app docker clean
install:
	python -m pip install -e ".[models,dev,app]"
test:
	pytest -q
lint:
	ruff check .
smoke:
	python -m book_trends.pipeline --generate-sample --models seasonal_naive
run:
	python -m book_trends.pipeline --generate-sample --tune
app:
	streamlit run app.py
docker: run
	docker build -t book-trends .
clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -not -path "./.venv*" -prune -exec rm -rf {} +
