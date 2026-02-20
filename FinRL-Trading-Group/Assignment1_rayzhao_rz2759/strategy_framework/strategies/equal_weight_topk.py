from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_framework.interfaces import StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class EqualWeightTopKConfig:
    top_n: int = 20
    rebalance_every_n_days: int = 20
    force_config_name: str | None = None


class EqualWeightTopKStrategy:
    name = "equal_weight_topk"

    def __init__(self, config: EqualWeightTopKConfig | None = None) -> None:
        self.config = config or EqualWeightTopKConfig()

    def generate_signals(self, selected_stocks_df: pd.DataFrame, ticker_sector, cc_ret_df) -> StrategySignals:
        cfg = self.config
        selected = selected_stocks_df.copy()
        selected["trade_date"] = pd.to_datetime(selected["trade_date"]).dt.normalize()
        selected = selected.dropna(subset=["tic", "predicted_return", "trade_date"])

        dates = pd.to_datetime(sorted(selected["trade_date"].unique()))
        all_tics = sorted(selected["tic"].unique())
        rebalance_dates = set(dates[:: cfg.rebalance_every_n_days])

        weights_dict = {}
        regime_rows = []
        current = {t: 0.0 for t in all_tics}

        for d in dates:
            if d in rebalance_dates or sum(current.values()) == 0.0:
                day = selected[selected["trade_date"] == d].sort_values("predicted_return", ascending=False).head(cfg.top_n)
                next_w = {t: 0.0 for t in all_tics}
                if not day.empty:
                    uniq = day["tic"].drop_duplicates().tolist()
                    w = 1.0 / len(uniq)
                    for t in uniq:
                        next_w[t] = w
                current = next_w

            weights_dict[d] = current.copy()
            regime_rows.append(
                {
                    "date": d,
                    "regime": 1,
                    "regime_label": "Neutral",
                    "market_vol": np.nan,
                    "market_mom": np.nan,
                    "alpha_power": np.nan,
                    "trade_executed": 1 if d in rebalance_dates else 0,
                }
            )

        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index").sort_index()
        weights_df.index.name = "date"
        regime_df = pd.DataFrame(regime_rows).set_index("date").sort_index()

        auto_name = f"eqw_topk{cfg.top_n}_reb{cfg.rebalance_every_n_days}"
        config_name = normalize_config_name(cfg.force_config_name) or auto_name

        return StrategySignals(weights_signal_df=weights_df, regime_df=regime_df, config_name=config_name)
