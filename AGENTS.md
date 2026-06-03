# AGENTS.md

## Cursor Cloud specific instructions

### Stack

Python package (`src/planning_detector/`), Streamlit UI (`streamlit_app.py`), configs in `configs/`, sample data in `data/sample/`. Legacy Colab notebook remains at repo root.

### Environment

```bash
python3 -m venv .venv   # requires python3.12-venv if ensurepip missing
source .venv/bin/activate
pip install -e ".[app,dev]"
export PLANNING_DATA_PATH=data/sample/planning_applications_sample.csv
```

### Common commands

| Task | Command |
|------|---------|
| Train | `planning-train` or `make train` |
| Tests | `make test` or `pytest -q` |
| Lint | `make lint` or `ruff check src tests` |
| Streamlit | `make app` or `streamlit run streamlit_app.py` |

### VM update script

```
test -d .venv || python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -e ".[app,dev]"
```

### Gotchas

- Default config uses `feature_set: early` to avoid processing-time leakage; `configs/retrospective.yaml` enables `full` features.
- Full Excel dataset is not in git; CI uses `data/sample/planning_applications_sample.csv`.
- Trained artifacts under `models/` are gitignored.
- One-time: `sudo apt install python3.12-venv` if venv creation fails.
