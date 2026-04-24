# P2-ETF-GRAPH-TRANSFORMER

**Graph Transformer – Global Attention across ETFs**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-GRAPH-TRANSFORMER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-GRAPH-TRANSFORMER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--graph--transformer--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-graph-transformer-results)

## Overview

`P2-ETF-GRAPH-TRANSFORMER` applies a **Graph Transformer** (TransformerConv) to daily ETF correlation graphs. Multi‑head self‑attention allows each ETF to attend to every other ETF directly, capturing long‑range dependencies that standard GCNs miss. The model predicts next‑day returns and ranks ETFs per universe.

## Methodology

- **Graph snapshots**: nodes = ETFs (features: recent returns + macro), edges = pairwise correlation > 0.5.
- **TransformerConv**: multiple layers of multi‑head attention.
- **Training**: full 2008–2026 dataset, one graph per trading day.
- **Inference**: latest graph snapshot → next‑day return predictions.

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
