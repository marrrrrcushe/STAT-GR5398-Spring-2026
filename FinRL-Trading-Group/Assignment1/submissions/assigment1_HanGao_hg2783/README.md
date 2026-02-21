# Regime-Aware Portfolio Backtest (README)

### Project Summary
Link to Medium: https://medium.com/p/257e00507a06?postPublishedType=initial

This project builds and backtests a systematic equity strategy and compares it to QQQ and SPY.
I evaluate three variants:
	1.	Portfolio_Base (No Regime)
	2.	Portfolio_Regime (Regime overlay using SPY MA200)
	3.	Portfolio_Regime_off_0.2 (More defensive regime overlay with risk-off exposure = 0.2)

All strategies share the same alpha engine (momentum selection + inverse-volatility weighting). The regime variants differ only in how they scale market exposure during risk-off periods.

### Key Data

Data Inputs
	•	output/step2/stock_selected.csv
Contains the selected stock universe by trade_date and tic.
	•	daily.csv
Contains daily prices and adjustment factor ajexdi used for split-adjusted close prices.
	•	Benchmarks: QQQ, SPY (downloaded via yfinance with auto_adjust=True).



### Return Definition

Let close_adj be split-adjusted close:

$$
P^{adj}{i,t} = \frac{P{i,t}}{ajexdi_{i,t}}
$$

Close-to-close return:

$$
r_{i,t} = \frac{P^{adj}{i,t}}{P^{adj}{i,t-1}} - 1
$$



### Strategy Details

1) Portfolio Construction (Base)

On each rebalance date, weights are assigned using inverse-volatility sizing:

$$
w_{i,t} \propto \frac{1}{\sigma_{i,t}}, \qquad \sum_i w_{i,t}=1
$$

where $$\sigma_{i,t}$$ is a rolling volatility estimate (e.g., 60 trading days). A weight cap is applied and the weights are renormalized.

2) Backtest Mechanics (Industry Standard)
	•	No look-ahead: weights are executed with a one-day lag:

$$
w^{exec}{i,t} = w^{signal}{i,t-1}
$$
	•	Transaction cost model:
Turnover is computed using drift-adjusted weights:

$$
\text{Turnover}t = \sum_i |w^{exec}{i,t} - w^{drift}_{i,t-1}|
$$

Cost:

$$
\text{Cost}_t = c \cdot \text{Turnover}_t
$$

where $$c=0.001$$ (10 bps).
	•	NAV calculation:

$$
\text{NAV}t = \text{NAV}{t-1}(1+r^{net}_t),\quad \text{NAV}_0=1
$$



### Regime Overlay (Portfolio_Regime / Portfolio_Regime_off_0.2)


Risk-off is triggered when SPY is below its 200-day moving average:

$$
\text{MA200}t = \frac{1}{200}\sum{k=0}^{199} S_{t-k}
$$

Risk-off condition:

$$
S_t < \text{MA200}_t
$$

Exposure Scaling

The regime overlay scales the base strategy return:

$$
r^{net,regime}_t = \alpha_t , r^{net,base}_t
$$

where:
	•	$$\alpha_t=1$$ in risk-on
	•	$$\alpha_t=\alpha_{off}$$ in risk-off
	•	Portfolio_Regime: baseline setting (less defensive)
	•	Portfolio_Regime_off_0.2: $$\alpha_{off}=0.2$$


### Results (Performance Metrics)

Metrics reported:
	•	Cumulative Return
	•	Annual Return (CAGR)
	•	Annual Volatility
	•	Sharpe Ratio (rf = 0)
	•	Max Drawdown

Strategy	Cumulative Return	Annual Return	Annual Volatility	Sharpe Ratio	Max Drawdown
Portfolio_Base	4.4693	0.2112	0.2637	0.8589	-0.4145
Portfolio_Regime	8.4385	0.2880	0.1956	1.3925	-0.1867
Portfolio_Regime_off_0.2	9.5034	0.3036	0.1879	1.5059	-0.1756
QQQ	4.1450	0.2028	0.2306	0.9166	-0.3512
SPY	2.4158	0.1486	0.1858	0.8388	-0.3372

Key takeaway: The regime overlay dramatically reduces drawdown and volatility while improving annual return, leading to much higher Sharpe ratios. The most defensive variant (off=0.2) achieves the best overall performance.



### How to Run
	1.	Make sure the following files exist:
	•	output/step2/stock_selected.csv
	•	daily.csv
	2.	Run the main notebook/script that:
	•	builds weights
	•	computes returns
	•	backtests (with costs)
	•	downloads benchmarks
	•	generates metrics + performance plot

Outputs:
	•	output/step2/custom_strategy_metrics*.csv
	•	output/step2/portfolio_comparison_chart*.png


