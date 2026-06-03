"""Streamlit demo for the Planning Application Issue Detector."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from planning_detector.config import REPO_ROOT, PipelineConfig, default_data_path
from planning_detector.labels import build_labels
from planning_detector.load import load_planning_data
from planning_detector.predict import PRED_COL, PROB_COL, load_artifact, score_dataframe
from planning_detector.train import METRICS_NAME, MODELS_DIR, train_pipeline

st.set_page_config(
    page_title="Planning Application Issue Detector",
    page_icon="🏗️",
    layout="wide",
)

st.title("Planning Application Issue Detector")
st.caption(
    "Flags Irish planning applications that may be problematic — "
    "long processing delays or refusals — using a reproducible ML pipeline."
)


def save_upload(uploaded) -> Path:
    dest = REPO_ROOT / "data" / "raw" / uploaded.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getvalue())
    return dest


@st.cache_data
def load_path_data(path_str: str) -> pd.DataFrame:
    config = PipelineConfig.load()
    return load_planning_data(Path(path_str), config)


def main() -> None:
    config = PipelineConfig.load()

    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

        if uploaded:
            data_path = save_upload(uploaded)
        else:
            data_path = default_data_path()
            st.caption(f"Using: `{data_path.name}`")

        st.divider()
        if st.button("Train model", type="primary"):
            with st.spinner("Training…"):
                _, metrics, meta = train_pipeline(data_path, config)
                st.success("Training complete.")
                st.json(metrics.to_dict())
                st.caption(f"Rows: {meta['n_rows']} | Features: {meta['feature_set']}")

    df = load_path_data(str(data_path))

    tab_overview, tab_scores, tab_about = st.tabs(["Overview", "Risk scores", "About"])

    with tab_overview:
        st.subheader("Dataset preview")
        st.dataframe(df.head(100), use_container_width=True)

        y = build_labels(df, config)
        col1, col2, col3 = st.columns(3)
        col1.metric("Applications", len(df))
        col2.metric("Problematic (label)", int(y.sum()))
        col3.metric("Problematic rate", f"{100 * y.mean():.1f}%")

        metrics_file = MODELS_DIR / METRICS_NAME
        if metrics_file.exists():
            st.subheader("Saved model metrics")
            st.json(json.loads(metrics_file.read_text()))

    with tab_scores:
        artifact_path = MODELS_DIR / "planning_detector.joblib"
        if not artifact_path.exists():
            st.info("Train a model from the sidebar to generate risk scores.")
        else:
            scored = score_dataframe(df, load_artifact(), config)
            threshold = st.slider("Flag if probability ≥", 0.0, 1.0, 0.5, 0.05)
            flagged = scored[scored[PROB_COL] >= threshold].sort_values(PROB_COL, ascending=False)

            st.metric("Flagged applications", len(flagged))
            show = [
                config.columns.planning_authority,
                config.columns.application_type,
                PROB_COL,
                PRED_COL,
            ]
            show = [c for c in show if c in flagged.columns]
            st.dataframe(flagged[show].head(200), use_container_width=True)

            st.download_button(
                "Download flagged rows (CSV)",
                flagged.to_csv(index=False).encode(),
                "flagged_applications.csv",
                "text/csv",
            )

    with tab_about:
        st.markdown(
            """
**Label modes** (`configs/default.yaml`): `delay`, `refusal`, `either`

**Feature sets**: `early` (recommended) or `full` (includes processing time — retrospective only)

```bash
pip install -e ".[app]"
planning-train
streamlit run streamlit_app.py
```
            """
        )


if __name__ == "__main__":
    main()
