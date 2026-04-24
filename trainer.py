"""
Main training script for Graph Transformer engine.
"""

import json
import pandas as pd
import numpy as np
import torch

import config
import data_manager
from graph_transformer_model import GTRunner
import push_results

def run_graph_transformer():
    print(f"=== P2-ETF-GRAPH-TRANSFORMER Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]
    macro = data_manager.prepare_macro(df_master)

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        graphs = data_manager.build_temporal_graph_sequence(returns, macro)
        if len(graphs) < config.MIN_OBSERVATIONS:
            continue

        in_channels = graphs[0].x.size(1)
        runner = GTRunner(
            in_channels=in_channels,
            hidden_channels=config.HIDDEN_CHANNELS,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            lr=config.LEARNING_RATE,
            seed=config.RANDOM_SEED
        )

        print(f"  Training Graph Transformer on {len(graphs)} snapshots...")
        runner.train_graphs(graphs, epochs=config.EPOCHS)

        preds = runner.predict_latest(graphs)
        universe_results = {}
        for i, ticker in enumerate(tickers):
            universe_results[ticker] = {
                "ticker": ticker,
                "forecast": float(preds[i])
            }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["forecast"], reverse=True)
        top_picks[universe_name] = [
            {k: v for k, v in d.items() if k != 'ticker'} | {"ticker": t}
            for t, d in sorted_tickers[:3]
        ]

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_graph_transformer()
