from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from strategy_framework.interfaces import StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class AlphaRPRegimeConfig:
    # Keep defaults aligned with the notebook alpha_rp implementation.
    top_n: int = 25
    vol_lookback_days: int = 63
    market_lookback_days: int = 21
    rebalance_every_n_days: int = 20
    rebalance_tolerance: float = 0.05

    score_power_risk_on: float = 1.35
    score_power_transition: float = 1.00
    score_power_risk_off: float = 0.75

    single_name_cap: float = 0.12
    sector_cap: float = 0.45

    risk_on_vol_th: float = 0.18
    risk_off_vol_th: float = 0.30
    risk_on_mom_th: float = -0.01
    risk_off_mom_th: float = -0.06

    force_config_name: str | None = None


class AlphaRPRegimeStrategy:
    """Alpha-tilted inverse-vol weighting with regime-aware alpha power."""

    name = "alpha_rp_regime"

    def __init__(self, config: AlphaRPRegimeConfig | None = None) -> None:
        self.config = config or AlphaRPRegimeConfig()
        self._regime_labels = {0: "RiskOn", 1: "Transition", 2: "RiskOff"}
        self._alpha_power_by_regime = {
            0: self.config.score_power_risk_on,
            1: self.config.score_power_transition,
            2: self.config.score_power_risk_off,
        }

    def _infer_market_regime(self, asof_date: pd.Timestamp, cc_ret_df: pd.DataFrame) -> tuple[int, float, float]:
        # Use only information strictly before asof_date to avoid look-ahead.
        hist = cc_ret_df.loc[cc_ret_df.index < asof_date].tail(self.config.market_lookback_days)
        market_ret = hist.mean(axis=1, skipna=True).dropna()
        if market_ret.empty:
            return 1, float("nan"), float("nan")

        market_vol = float(market_ret.std() * np.sqrt(252))
        market_mom = float((1.0 + market_ret).prod() - 1.0)

        if market_vol <= self.config.risk_on_vol_th and market_mom >= self.config.risk_on_mom_th:
            regime = 0
        elif market_vol <= self.config.risk_off_vol_th and market_mom >= self.config.risk_off_mom_th:
            regime = 1
        else:
            regime = 2
        return regime, market_vol, market_mom

    def _allocate_day(
        self,
        day_df: pd.DataFrame,
        asof_date: pd.Timestamp,
        cc_ret_df: pd.DataFrame,
        ticker_sector: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        cfg = self.config
        day_df = (
            day_df[["tic", "predicted_return"]]
            .dropna()
            .drop_duplicates(subset=["tic"])
            .sort_values("predicted_return", ascending=False)
            .head(cfg.top_n)
            .copy()
        )
        if day_df.empty:
            return {}, {"regime": 1, "market_vol": np.nan, "market_mom": np.nan, "alpha_power": cfg.score_power_transition}

        regime, market_vol, market_mom = self._infer_market_regime(asof_date, cc_ret_df)
        alpha_power = self._alpha_power_by_regime.get(regime, cfg.score_power_transition)

        tics = day_df["tic"].tolist()
        day_df["sector"] = day_df["tic"].map(ticker_sector).fillna(-1)

        # Volatility estimate must also be based on history before rebalance date.
        hist_ret = cc_ret_df.loc[cc_ret_df.index < asof_date, tics].tail(cfg.vol_lookback_days)
        vol = hist_ret.std(skipna=True).replace(0, np.nan)
        median_vol = vol.dropna().median()
        if pd.isna(median_vol):
            median_vol = 0.02
        vol = vol.fillna(median_vol).clip(lower=0.005)

        inv_vol = 1.0 / vol
        alpha = day_df["predicted_return"].clip(lower=0).to_numpy()
        if alpha.sum() <= 0:
            alpha = np.ones_like(alpha)
        alpha = np.power(alpha, alpha_power)

        raw = inv_vol.to_numpy() * alpha
        if raw.sum() <= 0:
            raw = np.ones_like(raw)
        base = raw / raw.sum()

        sectors = day_df["sector"].tolist()
        weights = {tic: 0.0 for tic in tics}
        sector_alloc: dict[float, float] = {}
        active = set(range(len(tics)))
        remaining = 1.0

        for _ in range(40):
            if remaining <= 1e-8 or not active:
                break

            base_sum = base[list(active)].sum()
            if base_sum <= 1e-12:
                break

            allocated = 0.0
            for i in list(active):
                tic = tics[i]
                sec = sectors[i]

                name_room = max(0.0, cfg.single_name_cap - weights[tic])
                sec_room = max(0.0, cfg.sector_cap - sector_alloc.get(sec, 0.0))
                room = min(name_room, sec_room)

                if room <= 1e-10:
                    active.discard(i)
                    continue

                target_add = remaining * (base[i] / base_sum)
                add = min(room, target_add)

                if add > 0:
                    weights[tic] += add
                    sector_alloc[sec] = sector_alloc.get(sec, 0.0) + add
                    allocated += add

                if name_room - add <= 1e-10:
                    active.discard(i)

            if allocated <= 1e-10:
                break
            remaining -= allocated

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        meta = {
            "regime": regime,
            "market_vol": market_vol,
            "market_mom": market_mom,
            "alpha_power": alpha_power,
        }
        return weights, meta

    def generate_signals(
        self,
        selected_stocks_df: pd.DataFrame,
        ticker_sector: Mapping[str, float],
        cc_ret_df: pd.DataFrame,
    ) -> StrategySignals:
        cfg = self.config
        selected = selected_stocks_df.copy()
        selected["trade_date"] = pd.to_datetime(selected["trade_date"]).dt.normalize()
        selected = selected.dropna(subset=["tic", "predicted_return", "trade_date"])

        portfolio_dates = pd.to_datetime(sorted(selected["trade_date"].unique()))
        all_stocks = sorted(selected["tic"].unique())
        rebalance_dates = set(portfolio_dates[:: cfg.rebalance_every_n_days])

        weights_dict: dict[pd.Timestamp, dict[str, float]] = {}
        regime_rows: list[dict[str, float]] = []
        current_weights = {tic: 0.0 for tic in all_stocks}

        for date in portfolio_dates:
            regime_today, market_vol_today, market_mom_today = self._infer_market_regime(date, cc_ret_df)
            alpha_power_today = self._alpha_power_by_regime.get(regime_today, cfg.score_power_transition)
            trade_executed = 0

            if date in rebalance_dates or sum(current_weights.values()) == 0.0:
                day_df = selected[selected["trade_date"] == date]
                target_weights, meta = self._allocate_day(
                    day_df=day_df,
                    asof_date=date,
                    cc_ret_df=cc_ret_df,
                    ticker_sector=ticker_sector,
                )

                target_full = {tic: target_weights.get(tic, 0.0) for tic in all_stocks}
                drift_l1 = sum(abs(current_weights[tic] - target_full[tic]) for tic in all_stocks)

                if sum(current_weights.values()) == 0.0 or drift_l1 >= cfg.rebalance_tolerance:
                    current_weights = target_full
                    trade_executed = 1

                regime_today = int(meta["regime"])
                market_vol_today = float(meta["market_vol"])
                market_mom_today = float(meta["market_mom"])
                alpha_power_today = float(meta["alpha_power"])

            weights_dict[date] = current_weights.copy()
            regime_rows.append(
                {
                    "date": date,
                    "regime": regime_today,
                    "regime_label": self._regime_labels.get(regime_today, "Transition"),
                    "market_vol": market_vol_today,
                    "market_mom": market_mom_today,
                    "alpha_power": alpha_power_today,
                    "trade_executed": trade_executed,
                }
            )

        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index").sort_index()
        weights_df.index.name = "date"
        regime_df = pd.DataFrame(regime_rows).set_index("date").sort_index()

        auto_name = (
            f"alpha_rp_regime_topk{cfg.top_n}_vol{cfg.vol_lookback_days}"
            f"_cap{int(cfg.single_name_cap * 100)}_sec{int(cfg.sector_cap * 100)}"
            f"_reb{cfg.rebalance_every_n_days}_tol{int(cfg.rebalance_tolerance * 100)}"
        )
        config_name = normalize_config_name(cfg.force_config_name) or auto_name

        return StrategySignals(
            weights_signal_df=weights_df,
            regime_df=regime_df,
            config_name=config_name,
            metadata={
                "strategy": self.name,
                "top_n": cfg.top_n,
                "rebalance_every_n_days": cfg.rebalance_every_n_days,
            },
        )
