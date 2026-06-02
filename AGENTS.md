# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

Single-product **Jupyter/Colab ML notebook** (`Irish_Planning_Data_Ireland_Cloud_Version_No_API.ipynb`) for predicting **problematic Irish planning applications** (primarily processing time &gt; 180 days). There is no web app, API, Docker stack, or automated test/lint configuration.

### One-time VM prerequisites

The base image may lack `python3-venv`. If `python3 -m venv .venv` fails, run once (outside the update script):

```bash
sudo apt-get update && sudo apt-get install -y python3.12-venv
```

### Python environment

- Use the project virtualenv: `/workspace/.venv`
- Activate: `source .venv/bin/activate` or prefix commands with `.venv/bin/`
- Core dependencies are installed via the VM **update script** (see SetupVmEnvironment). They match the notebook’s `!pip install` cells: `openpyxl`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `nltk`, `scikit-learn`, `xgboost`, `jupyter`, `ipykernel`, `psutil`.

**Optional (heavy):** `transformers` and `torch` for T5 summarization cells. Install when needed:

```bash
.venv/bin/pip install transformers torch
```

After install, download NLTK corpora once per environment:

```bash
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dataset (required for full notebook run)

The real dataset is **not in git**: `Tableau_Ready_Planning_Applications_With_Street_Town_Cleaned.xlsx`.

- **Colab path:** `/content/drive/MyDrive/Planning Application Database/...`
- **Local path:** place the file in `/workspace/` (early notebook cells use the filename in the working directory).

Many cells import `google.colab` and mount Drive; they must be skipped or adapted for local runs.

### Running

| Task | Command |
|------|---------|
| Jupyter Lab | `cd /workspace && .venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8888` (use tmux for long-running server) |
| Quick pipeline smoke test | Run EDA + `ProcessingTime` + `Problematic` label + sklearn/XGBoost training against a local `.xlsx` (see README methodology) |
| Lint | Not configured |
| Tests | Not configured |

### Gotchas

- **No `requirements.txt` in repo** — dependency list is inferred from notebook `!pip install` lines.
- **Colab-only cells** will fail under plain Jupyter/`nbconvert --execute` unless edited.
- **Summarization** needs network access to download Hugging Face `t5-small`.
- High accuracy on a small synthetic sample is expected when `ProcessingTime` is a feature and the target is derived from it (&gt; 180 days); the full notebook uses richer feature engineering on real data.
