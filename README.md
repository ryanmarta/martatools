# NEXUS TITAN – Quant Options & Scanner Dashboard

Production-grade Streamlit app for options pricing, Greeks, edge/PoP analysis, backtesting, and Ryan Model market scanning (S&P 100+). Data is sourced live from `yfinance` (no synthetic or Alpha Vantage).

## Features
- Pricing Lab: BSM + Heston proxy + Monte Carlo consensus, 3rd-order Greeks, PoP, edge, scenario P/L, optimal edge picks across the chain, and rolling walk-forward backtest (ATM, tenor-matched).
- Dashboard: Price intel with regime metrics and candlesticks.
- Hunter (AI Quant): Vol squeeze + volume velocity + relative strength + Monte Carlo.
- Nexus Scanner (Ryan Model): S&P 100+ squeeze/trap/RS/confidence scanning with long/short decks.

## Requirements
- Python 3.9+ recommended
- Dependencies (installed): `streamlit`, `yfinance`, `pandas`, `numpy`, `plotly`, `scipy`, `scikit-learn`, `requests`

## Quickstart
```bash
pip install --upgrade streamlit yfinance pandas numpy plotly scipy scikit-learn requests
streamlit run app.py
```

## Usage Tips
- Pricing Lab: pick ticker/expiry/strike/type → run → review edge, PoP, backtests, and optimal picks table.
- Hunter: configure windows/thresholds; Monte Carlo included for intuition.
- Nexus Scanner (Future Module): run the Ryan Model scan; long/short tables plus decoder key.

## MIT License
Copyright (c) 2025 Ryan M.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

