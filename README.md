# European Power Market — Streamlit Demo

This repository contains a simple Streamlit app that models a synthetic European power market.

Files:
- streamlit_dashboard.py — main Streamlit app
- requirements.txt — Python dependencies

Quick start

1. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_dashboard.py
```

Notes

- The model is intentionally simple: no transmission constraints, no unit commitment, and no market bidding dynamics beyond a merit-order dispatch with fixed marginal costs.
- To extend: add per-unit commitment, variable marginal costs, imports/exports between countries, or use real input data.
