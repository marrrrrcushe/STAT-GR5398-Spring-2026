from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from strategy_framework.interfaces import StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class AdaptiveRegimeCorrRPRankAlphaConfig:
    top_n: int = 25
    vol_lookback_days: int = 63
    corr_lookback_days: int = 63
    market_lookback_days: int = 21
    regime_ref_lookback_days: int = 252
    regime_ewma_span: int = 63
    rebalance_every_n_days: int = 20
    rebalance_tolerance: float = 0.05

    rank_power_risk_on: float = 1.25
    rank_power_transition: float = 1.00
    rank_power_risk_off: float = 0.85

    single_name_cap: float = 0.12
    sector_cap: float = 0.45

    corr_penalty_threshold: float = 0.80
    corr_penalty_strength: float = 1.00

    risk_on_z_th: float = -0.50
    risk_off_z_th: float = 1.50

    force_config_name: str | None = None


class AdaptiveRegimeCorrRPRankAlphaStrategy:
    """
    Dynamic-regime + correlation-penalized inverse-vol + rank-normalized alpha.
    """

    name = "AdaptiveRegime_CorrRP_RankAlpha"

    def __init__(self, config: AdaptiveRegimeCorrRPRankAlphaConfig | None = None) -> None:
        self.config = config or AdaptiveRegimeCorrRPRankAlphaConfig()
        self._regime_labels = {0: "RiskOn", 1: "Transition", 2: "RiskOff"}
        self._rank_power_by_regime = {
            0: self.config.rank_power_risk_on,
            1: self.config.rank_power_transition,
            2: self.config.rank_power_risk_off,
        }

    def _infer_market_regime(self, asof_date: pd.Timestamp, cc_ret_df: pd.DataFrame) -> tuple[int, float, float, float]:
        cfg = self.config
        min_periods = max(5, cfg.market_lookback_days // 2)

        hist = cc_ret_df.loc[cc_ret_df.index < asof_date]
        market_ret = hist.mean(axis=1, skipna=True).dropna()
        if len(market_ret) < min_periods:
            return 1, float("nan"), float("nan"), float("nan")

        # 1) Build annualized rolling vol (history-only), then log-transform it to reduce right-tail skew.
        rolling_vol = (
            market_ret.rolling(cfg.market_lookback_days, min_periods=min_periods).std() * np.sqrt(252)
        ).dropna()
        if rolling_vol.empty:
            return 1, float("nan"), float("nan"), float("nan")

        current_vol = float(rolling_vol.iloc[-1])
        log_vol = np.log(rolling_vol.clip(lower=1e-8))
        ref_log_vol = log_vol.tail(cfg.regime_ref_lookback_days)
        if ref_log_vol.empty:
            return 1, current_vol, float("nan"), float("nan")

        # 2) Use EWMA mean/std on log-vol to avoid SMA ghost effect in regime baseline.
        ewm_mean = ref_log_vol.ewm(span=cfg.regime_ewma_span, adjust=False).mean()
        ewm_std = ref_log_vol.ewm(span=cfg.regime_ewma_span, adjust=False).std(bias=False)
        ref_mean = float(ewm_mean.iloc[-1])
        ref_std = float(ewm_std.iloc[-1]) if not pd.isna(ewm_std.iloc[-1]) else float(ref_log_vol.std())
        if ref_std <= 1e-12 or pd.isna(ref_std):
            vol_z = 0.0
        else:
            vol_z = float((ref_log_vol.iloc[-1] - ref_mean) / ref_std)

        vol_pct = float((ref_log_vol <= ref_log_vol.iloc[-1]).mean())

        if vol_z <= cfg.risk_on_z_th:
            regime = 0
        elif vol_z >= cfg.risk_off_z_th:
            regime = 2
        else:
            regime = 1

        return regime, current_vol, vol_pct, vol_z

    def _correlation_penalty(self, hist_ret: pd.DataFrame) -> pd.Series:
        cfg = self.config
        if hist_ret.empty:
            return pd.Series(dtype=float)

        corr = hist_ret.corr().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if corr.empty:
            return pd.Series(dtype=float)

        for col in corr.columns:
            corr.loc[col, col] = 0.0

        excess_corr = (corr - cfg.corr_penalty_threshold).clip(lower=0.0)
        avg_excess = excess_corr.mean(axis=1)
        penalty = 1.0 / (1.0 + cfg.corr_penalty_strength * avg_excess)
        return penalty.clip(lower=0.25, upper=1.0)

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
            return {}, {
                "regime": 1,
                "market_vol": np.nan,
                "market_vol_pct": np.nan,
                "market_vol_z": np.nan,
                "rank_power": cfg.rank_power_transition,
            }

        regime, market_vol, market_vol_pct, market_vol_z = self._infer_market_regime(asof_date, cc_ret_df)
        rank_power = self._rank_power_by_regime.get(regime, cfg.rank_power_transition)

        tics = day_df["tic"].tolist()
        day_df["sector"] = day_df["tic"].map(ticker_sector).fillna(-1)

        hist_ret = cc_ret_df.loc[cc_ret_df.index < asof_date, tics]
        vol_hist = hist_ret.tail(cfg.vol_lookback_days)
        corr_hist = hist_ret.tail(cfg.corr_lookback_days)

        vol = vol_hist.std(skipna=True).replace(0, np.nan)
        median_vol = vol.dropna().median()
        if pd.isna(median_vol):
            median_vol = 0.02
        vol = vol.fillna(median_vol).clip(lower=0.005)

        rank_pct = day_df["predicted_return"].rank(method="average", pct=True).clip(lower=1e-6)
        rank_alpha = np.power(rank_pct.to_numpy(), rank_power)
        inv_vol = 1.0 / vol

        corr_penalty = self._correlation_penalty(corr_hist).reindex(tics).fillna(1.0)

        raw = inv_vol.to_numpy() * corr_penalty.to_numpy() * rank_alpha
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
            "market_vol_pct": market_vol_pct,
            "market_vol_z": market_vol_z,
            "rank_power": rank_power,
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
            regime_today, market_vol_today, market_vol_pct_today, market_vol_z_today = self._infer_market_regime(
                date, cc_ret_df
            )
            rank_power_today = self._rank_power_by_regime.get(regime_today, cfg.rank_power_transition)
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
                market_vol_pct_today = float(meta["market_vol_pct"])
                market_vol_z_today = float(meta["market_vol_z"])
                rank_power_today = float(meta["rank_power"])

            weights_dict[date] = current_weights.copy()
            regime_rows.append(
                {
                    "date": date,
                    "regime": regime_today,
                    "regime_label": self._regime_labels.get(regime_today, "Transition"),
                    "market_vol": market_vol_today,
                    "market_vol_pct": market_vol_pct_today,
                    "market_vol_z": market_vol_z_today,
                    "rank_power": rank_power_today,
                    "trade_executed": trade_executed,
                }
            )

        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index").sort_index()
        weights_df.index.name = "date"
        regime_df = pd.DataFrame(regime_rows).set_index("date").sort_index()

        auto_name = (
            f"adaptive_regime_corrrp_rankalpha_topk{cfg.top_n}_vol{cfg.vol_lookback_days}"
            f"_corr{cfg.corr_lookback_days}_cap{int(cfg.single_name_cap * 100)}"
            f"_sec{int(cfg.sector_cap * 100)}_reb{cfg.rebalance_every_n_days}"
            f"_tol{int(cfg.rebalance_tolerance * 100)}"
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
