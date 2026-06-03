.PHONY: install train app test lint

install:
	pip install -e ".[app,dev]"

train:
	PLANNING_DATA_PATH=data/sample/planning_applications_sample.csv planning-train

test:
	PLANNING_DATA_PATH=data/sample/planning_applications_sample.csv pytest -q

lint:
	ruff check src tests streamlit_app.py

app:
	streamlit run streamlit_app.py
