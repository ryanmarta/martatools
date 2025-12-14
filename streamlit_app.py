import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import norm
import streamlit as st
import yfinance as yf
from scipy.interpolate import CubicSpline
import requests  # kept for Scanner batch download

# Optional GARCH dependency for Quant Stack Layer 3
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# ------------------------------------------------------------------------------
# 1. CONFIGURATION & THEME
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Marta Tools",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")

st.markdown(
    """
    <style>
        /* GLOBAL THEME: Slate & White (Institutional) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        
        .stApp { background-color: #F8FAFC; color: #1E293B; font-family: 'Inter', sans-serif; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
        
        /* METRIC CARDS */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E1;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricValue"] { color: #0F172A !important; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
        div[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.75rem; letter-spacing: 1px; font-weight: 600; }
        
        /* ALERTS */
        .signal-box {
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-family: 'JetBrains Mono', monospace;
            border-left: 4px solid #64748B;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .signal-buy { border-left-color: #10B981; background: #ECFDF5; color: #064E3B; }
        .signal-sell { border-left-color: #EF4444; background: #FEF2F2; color: #991B1B; }
        
        /* DATAFRAMES */
        .stDataFrame { border: 1px solid #E2E8F0; border-radius: 6px; }
        
        /* HEADERS */
        h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; color: #0F172A; }
        
        /* EDUCATIONAL FOOTER */
        .edu-footer {
            background-color: #F1F5F9;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2563EB;
            margin-top: 30px;
            color: #334155;
        }
        
        /* MOBILE IMPROVEMENTS */
        @media (max-width: 768px) {
            /* Make sidebar narrower on mobile */
            section[data-testid="stSidebar"] {
                width: 200px !important;
                min-width: 200px !important;
            }
            section[data-testid="stSidebar"] > div {
                padding: 0.5rem !important;
            }
        }
        
        /* Better radio button styling */
        .stRadio > div {
            gap: 0.25rem !important;
        }
        .stRadio > div > label {
            background: #F1F5F9 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            margin: 2px 0 !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
            border: 2px solid transparent !important;
        }
        .stRadio > div > label:hover {
            background: #E2E8F0 !important;
        }
        .stRadio > div > label[data-checked="true"],
        .stRadio > div > label:has(input:checked) {
            background: #2563EB !important;
            color: white !important;
            border-color: #1D4ED8 !important;
        }
        /* Hide default radio circles */
        .stRadio > div > label > div:first-child {
            display: none !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. CORE MATH & UTILS
# ------------------------------------------------------------------------------


class QuantMath:
    @staticmethod
    def norm_cdf(x: float) -> float:
        return stats.norm.cdf(x)

    @staticmethod
    def norm_pdf(x: float) -> float:
        return stats.norm.pdf(x)

    @staticmethod
    def safe_div(n, d, default=0.0):
        return n / d if d != 0 and not np.isnan(d) else default


@dataclass
class GreekVector:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    charm: float
    speed: float
    zomma: float
    color: float
    ultima: float


# ------------------------------------------------------------------------------
# 3. HYDRA DATA ENGINE (FAIL-SAFE)
# ------------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def _fetch_spot_cached(ticker: str) -> Tuple[float, str, List[str]]:
    """Cached spot price fetcher for Streamlit Cloud."""
    debug_log = []
    
    # Method 1: yf.download (most reliable on Streamlit Cloud)
    try:
        df = yf.download(ticker, period="5d", progress=False)
        if df is not None and not df.empty:
            # Handle both single ticker and multi-ticker column formats
            if isinstance(df.columns, pd.MultiIndex):
                price = float(df["Close"][ticker].iloc[-1])
            else:
                price = float(df["Close"].iloc[-1])
            if price > 0:
                return price, "Last Close", debug_log
        debug_log.append("download: empty df")
    except Exception as e:
        debug_log.append(f"download: {str(e)[:80]}")

    # Method 2: Ticker.history (good fallback)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if price > 0:
                return price, "History", debug_log
        debug_log.append("history: empty")
    except Exception as e:
        debug_log.append(f"history: {str(e)[:80]}")

    # Method 3: Ticker.fast_info
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.last_price
        if price and price > 0:
            return float(price), "Live", debug_log
        debug_log.append("fast_info: no price")
    except Exception as e:
        debug_log.append(f"fast_info: {str(e)[:80]}")

    # Method 4: Ticker.info dict
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if info:
            price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice")
            if price and price > 0:
                return float(price), "Info", debug_log
        debug_log.append("info: no price keys")
    except Exception as e:
        debug_log.append(f"info: {str(e)[:80]}")

    return None, "NO_DATA", debug_log


@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def _fetch_history_cached(ticker: str) -> pd.DataFrame:
    """Cached history fetcher for Streamlit Cloud."""
    # Method 1: yf.download
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df is not None and not df.empty:
            # Handle multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            return df
    except Exception:
        pass

    # Method 2: Ticker.history
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()


class HydraEngine:
    """
    Robust Data Router for Streamlit Cloud.
    Uses cached yfinance calls to avoid rate limiting.
    """

    def __init__(self):
        self.debug_log = []

    def get_spot(self, ticker: str) -> Tuple[float, str]:
        self.debug_log = []
        
        # Use cached function
        price, source, debug = _fetch_spot_cached(ticker)
        self.debug_log = debug
        return price, source

    def get_history(self, ticker: str) -> pd.DataFrame:
        return _fetch_history_cached(ticker)
    
    def get_debug_info(self) -> str:
        return " | ".join(self.debug_log) if self.debug_log else "No debug info"

    def get_chain(self, ticker: str, expiry: str, spot: float) -> Tuple[pd.DataFrame, bool]:
        try:
            t = yf.Ticker(ticker)
            chain = t.option_chain(expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["type"] = "call"
            puts["type"] = "put"
            df = pd.concat([calls, puts], ignore_index=True)
            df["mid"] = (df["bid"] + df["ask"]) / 2
            df.loc[df["mid"] == 0, "mid"] = df.loc[df["mid"] == 0, "lastPrice"]
            df = df[df["mid"] > 0.01].copy()
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df = df.dropna(subset=["strike", "mid", "impliedVolatility"])
            return df, False
        except Exception:
            return pd.DataFrame(), True

    def get_expirations(self, ticker: str) -> List[str]:
        try:
            options = list(yf.Ticker(ticker).options)
            return options or []
        except Exception:
            return []


# ------------------------------------------------------------------------------
# 3.5 QUANT STACK ENGINE (Kalman, Macro Regime, GARCH)
# ------------------------------------------------------------------------------


class SimpleKalmanFilter:
    """1D Kalman Filter for price signal extraction."""
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def input_latest_noisy_measurement(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        return self.posteri_estimate


class QuantStackEngine:
    """
    3-Layer Quant Analysis:
    1. Direction (Four Horsemen Macro Regime)
    2. Valuation (Kalman Filter)
    3. Sizing (GARCH Volatility)
    """
    MACRO_TICKERS = ['JNK', 'LQD', 'XLY', 'XLP', 'QQQ', 'TLT', 'SPY', 'GLD']
    
    def __init__(self, target_ticker, history_df):
        self.target = target_ticker
        self.history = history_df
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_macro_data(_self):
        """Fetch macro indicator data for regime detection."""
        try:
            data = yf.download(_self.MACRO_TICKERS, period="1y", progress=False)
            
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Extract just the 'Close' prices for all tickers
                if 'Close' in data.columns.get_level_values(0):
                    closes = data['Close']
                else:
                    return pd.DataFrame()
            else:
                # Single ticker or already flattened
                if 'Close' in data.columns:
                    closes = data[['Close']]
                else:
                    return pd.DataFrame()
            
            # Ensure we have a DataFrame, not a Series
            if isinstance(closes, pd.Series):
                closes = closes.to_frame()
            
            closes = closes.ffill().dropna()
            return closes
        except Exception:
            return pd.DataFrame()

    def run_layer1_regime(self):
        """Layer 1: Direction (Four Horsemen Macro Regime)"""
        closes = self.fetch_macro_data()
        if closes.empty:
            return "NEUTRAL", 0.0, pd.DataFrame()

        factors = pd.DataFrame(index=closes.index)
        needed = set(self.MACRO_TICKERS)
        existing = set(closes.columns)
        if not needed.issubset(existing):
            return "NEUTRAL", 0.0, pd.DataFrame()

        factors['Credit'] = closes['JNK'] / closes['LQD']
        factors['Consumer'] = closes['XLY'] / closes['XLP']
        factors['Growth'] = closes['QQQ'] / closes['TLT']
        factors['Fear'] = closes['SPY'] / closes['GLD']

        z_scores = pd.DataFrame(index=factors.index)
        for col in factors.columns:
            roll_mean = factors[col].rolling(60).mean()
            roll_std = factors[col].rolling(60).std()
            z_scores[col] = (factors[col] - roll_mean) / roll_std

        z_scores['Macro_Score'] = z_scores.sum(axis=1)
        current_score = z_scores['Macro_Score'].iloc[-1]
        
        regime = "NEUTRAL"
        if current_score > 1.0: 
            regime = "BULL"
        elif current_score < -1.0: 
            regime = "BEAR"
        
        return regime, current_score, z_scores

    def run_layer2_kalman(self):
        """Layer 2: Valuation (Kalman Filter True Value)"""
        if self.history.empty:
            return None
            
        prices = self.history['Close'].values
        kf = SimpleKalmanFilter(1e-5, 1e-1, 1.0)
        kf.posteri_estimate = prices[0]
        
        kalman_vals = []
        for p in prices:
            kalman_vals.append(kf.input_latest_noisy_measurement(p))
            
        df = self.history.copy()
        df['Kalman'] = kalman_vals
        df['Residual'] = df['Close'] - df['Kalman']
        df['Res_Mean'] = df['Residual'].rolling(20).mean()
        df['Res_Std'] = df['Residual'].rolling(20).std()
        df['Kalman_Z'] = (df['Residual'] - df['Res_Mean']) / df['Res_Std']
        
        return df

    def run_layer3_garch(self):
        """Layer 3: Sizing (GARCH Volatility)"""
        if not ARCH_AVAILABLE:
            return 1.0, 1.0, "N/A (arch not installed)"
            
        if self.history.empty or len(self.history) < 100:
            return 1.0, 1.0, "Insufficient data"
            
        returns = 100 * self.history['Close'].pct_change().dropna()
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            current_vol = res.conditional_volatility.iloc[-1]
            cond_vol = res.conditional_volatility
            vol_mean_60 = cond_vol.rolling(60).mean().iloc[-1]
            vol_ratio = current_vol / vol_mean_60 if vol_mean_60 > 0 else 1.0
            
            sizing = 1.0
            status = "NORMAL"
            if vol_ratio > 1.2:
                sizing = 0.5
                status = "HIGH VOL (Size Down)"
            elif vol_ratio < 0.8:
                sizing = 1.2
                status = "LOW VOL (Size Up)"
                
            return sizing, vol_ratio, status
        except:
            return 1.0, 1.0, "GARCH Error"


# ------------------------------------------------------------------------------
# 4. VOLATILITY & SURFACE
# ------------------------------------------------------------------------------


class SurfaceEngine:
    def __init__(self, chain_df):
        self.valid = False
        self.model = None
        self._fit(chain_df)

    def _fit(self, df):
        if df is None or df.empty:
            return
        subset = df[(df["impliedVolatility"] > 0.01) & (df["impliedVolatility"] < 5.0)]
        agg = subset.groupby("strike")["impliedVolatility"].mean().reset_index().sort_values("strike")
        if len(agg) < 4:
            return
        try:
            self.model = CubicSpline(agg["strike"], agg["impliedVolatility"], bc_type="natural")
            self.valid = True
        except Exception:
            pass

    def get_iv(self, strike):
        return float(max(0.01, self.model(strike))) if (self.valid and self.model) else None


# ------------------------------------------------------------------------------
# 5. PRICING KERNEL (TRINITY)
# ------------------------------------------------------------------------------


class PricingKernel:
    def __init__(self, S, K, T, r, sigma, type_):
        self.S, self.K, self.T = float(S), float(K), max(float(T), 0.001)
        self.r, self.v, self.type = float(r), max(float(sigma), 0.001), type_.lower()
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.v**2) * self.T) / (self.v * np.sqrt(self.T))
        self.d2 = self.d1 - self.v * np.sqrt(self.T)

    def price_bsm(self) -> float:
        if self.type == "call":
            return self.S * QuantMath.norm_cdf(self.d1) - self.K * np.exp(-self.r * self.T) * QuantMath.norm_cdf(self.d2)
        return self.K * np.exp(-self.r * self.T) * QuantMath.norm_cdf(-self.d2) - self.S * QuantMath.norm_cdf(-self.d1)

    def price_mc(self, sims=50000) -> float:
        N = int(sims / 2)
        z = np.random.standard_normal(N)
        z = np.concatenate((z, -z))
        drift = (self.r - 0.5 * self.v**2) * self.T
        diffusion = self.v * np.sqrt(self.T) * z
        ST = self.S * np.exp(drift + diffusion)
        payoff = np.maximum(ST - self.K, 0) if self.type == "call" else np.maximum(self.K - ST, 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def price_heston(self) -> float:
        # Lewis Proxy
        moneyness = np.log(self.S / self.K)
        rho, sigma_v = -0.7, 0.3
        skew_adj = (rho * sigma_v * moneyness) / (2 * self.v)
        smile_adj = (sigma_v**2 * moneyness**2) / (12 * self.v**2)
        adj_vol = self.v * (1 + skew_adj + smile_adj)
        return PricingKernel(self.S, self.K, self.T, self.r, adj_vol, self.type).price_bsm()

    def price_heston_proxy(self) -> float:
        """Alias for backwards compatibility with earlier UI text."""
        return self.price_heston()

    def get_greeks(self) -> GreekVector:
        pdf, cdf = QuantMath.norm_pdf(self.d1), QuantMath.norm_cdf(self.d1)
        sqT = np.sqrt(self.T)

        delta = cdf if self.type == "call" else cdf - 1
        gamma = pdf / (self.S * self.v * sqT)
        vega = (self.S * pdf * sqT) / 100.0
        theta = (-self.S * pdf * self.v / (2 * sqT)) / 365.0
        rho = (self.K * self.T * np.exp(-self.r * self.T) * QuantMath.norm_cdf(self.d2)) / 100.0

        # 2nd/3rd Order
        vanna = -pdf * self.d2 / self.v
        volga = vega * self.d1 * self.d2 / self.v
        charm = -pdf * (2 * self.r * self.T - self.d2 * self.v * sqT) / (2 * self.T * self.v * sqT)
        speed = -gamma / self.S * (self.d1 / (self.v * sqT) + 1)
        zomma = gamma * (self.d1 * self.d2 - 1) / self.v
        color = -gamma / (2 * self.T) * (1 + self.d1 / (self.v * sqT) * (2 * self.r * self.T - self.d2 * self.v * sqT) + self.d1 * self.d2)
        ultima = -vega * (self.d1 * self.d2 * (1 - self.d1 * self.d2) + self.d1**2 + self.d2**2) / (self.v**2)

        return GreekVector(delta, gamma, vega, theta, rho, vanna, volga, charm, speed, zomma, color, ultima)


# ------------------------------------------------------------------------------
# 6. THE HUNTER: RESTORED SCANNER ENGINE
# ------------------------------------------------------------------------------


class ScannerEngine:
    WATCHLIST = [
        "NVDA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AVGO",
        "ORCL",
        "ADBE",
        "CRM",
        "AMD",
        "QCOM",
        "TXN",
        "JPM",
        "BAC",
        "V",
        "MA",
        "WFC",
        "GS",
        "MS",
        "AXP",
        "BLK",
        "C",
        "PYPL",
        "WMT",
        "COST",
        "PG",
        "HD",
        "KO",
        "PEP",
        "MCD",
        "DIS",
        "NKE",
        "SBUX",
        "LLY",
        "UNH",
        "JNJ",
        "MRK",
        "ABBV",
        "PFE",
        "AMGN",
        "GILD",
        "CAT",
        "DE",
        "HON",
        "GE",
        "BA",
        "LMT",
        "RTX",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "PLTR",
        "DKNG",
        "ROKU",
        "SQ",
        "COIN",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
    ]

    @staticmethod
    def run_batch_scan(window, thresh):
        """
        Executes the specific Ryan Model logic:
        1. Trend Bias
        2. Squeeze Detection
        3. Volume Velocity
        4. Trap Detection (Health)
        5. Confidence Scoring
        """

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        # Batch Download for speed (Stealth Headers)
        try:
            tickers_str = " ".join(ScannerEngine.WATCHLIST)
            data = yf.download(tickers_str, period="1y", group_by="ticker", progress=False, session=session)
        except Exception:
            return pd.DataFrame()

        results = []

        for symbol in ScannerEngine.WATCHLIST:
            try:
                if symbol not in data.columns.levels[0]:
                    continue
                df = data[symbol].dropna()
                if len(df) < 90:
                    continue

                close = df["Close"]
                volume = df["Volume"]

                # Squeeze
                sma = close.rolling(window).mean()
                std = close.rolling(window).std()
                upper = sma + (2 * std)
                lower = sma - (2 * std)
                bw = (upper - lower) / sma
                current_bw = bw.iloc[-1]
                is_squeeze = current_bw < thresh

                # Trend
                curr_p = close.iloc[-1]
                trend = "BULLISH" if curr_p > sma.iloc[-1] else "BEARISH"

                # Momentum (EMA)
                ema_9 = close.ewm(span=9, adjust=False).mean()
                ema_21 = close.ewm(span=21, adjust=False).mean()
                mom = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"

                # Volume Velocity
                v3 = volume.rolling(3).mean().iloc[-1]
                v30 = volume.rolling(30).mean().iloc[-1]
                v60 = volume.rolling(60).mean().iloc[-1]
                v_score = 0
                if v3 > v30:
                    v_score += 1
                if v3 > v60:
                    v_score += 1
                vol_status = "High Vel" if v_score >= 2 else "Low Vel"

                # Trap / Health
                high = df["High"].iloc[-1]
                low = df["Low"].iloc[-1]
                rng = high - low
                pos = (curr_p - low) / rng if rng > 0 else 0.5

                if pos > 0.8:
                    health = "POWER"
                elif pos < 0.2:
                    health = "WEAK"
                elif pos < 0.5 and trend == "BULLISH":
                    health = "FADING (Trap)"
                else:
                    health = "NEUTRAL"

                rs_rating = 0.0  # Placeholder if SPY not strictly aligned in index

                # --- SCORING ENGINE ---
                confidence = 0
                if trend == "BULLISH" and mom == "BULLISH":
                    confidence += 35
                elif trend == "BEARISH" and mom == "BEARISH":
                    confidence += 35
                else:
                    confidence += 15

                if is_squeeze:
                    confidence += 25
                if vol_status == "High Vel":
                    confidence += 20
                if health == "POWER" and trend == "BULLISH":
                    confidence += 20
                if health == "WEAK" and trend == "BEARISH":
                    confidence += 20

                # Penalties
                if health == "FADING (Trap)":
                    confidence -= 20

                confidence = min(max(confidence, 0), 100)

                # Action
                action = "WAIT"
                if confidence >= 80:
                    if trend == "BULLISH":
                        action = "✅ BUY"
                    else:
                        action = "🔻 SHORT"

                results.append(
                    {
                        "Ticker": symbol,
                        "Action": action,
                        "Confidence": confidence,
                        "Price": curr_p,
                        "Trend": trend,
                        "Momentum": mom,
                        "Squeeze": "COILED" if is_squeeze else "LOOSE",
                        "Health": health,
                        "Bandwidth": current_bw,
                        "Vol_Vel": vol_status,
                    }
                )

            except Exception:
                continue

        return pd.DataFrame(results)


# ------------------------------------------------------------------------------
# 7. MAIN UI LAYOUT
# ------------------------------------------------------------------------------


def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🛠️ Marta Tools")
        
        mode = st.radio("Select Module", ["📺 Dashboard", "💎 Options", "🎯 Sniper", "🦅 Hunter", "🔍 Opt Hunt", "📊 ETF Hunt", "📊 ETF Opts", "🩳 Shorties", "👃 Pickers"], label_visibility="collapsed")

        if mode == "🎯 Sniper":
            st.markdown("---")
            ticker = st.text_input("TICKER", value="TSLA").upper()
            rf = st.number_input("RISK FREE (%)", value=4.5)
            cap = st.number_input("CAPITAL ($)", value=100000)
            kelly = st.slider("KELLY FACTOR", 0.1, 1.0, 0.5)
        
        st.markdown("---")
        st.caption("Tap ✕ or swipe to close")

    # --- SHARED INIT (only for Sniper mode now) ---
    if mode == "🎯 Sniper":
        hydra = HydraEngine()
        with st.spinner(f"Connecting to {ticker}..."):
            spot, src = hydra.get_spot(ticker)
            hist = hydra.get_history(ticker)

        if spot is None:
            st.error("❌ Unable to retrieve market data. Please check the ticker symbol or your internet connection.")
            with st.expander("🔧 Debug Info"):
                st.code(hydra.get_debug_info())
                st.info("Try: pip install --upgrade yfinance")
            return
        if hist is None or hist.empty:
            st.warning("⚠️ Limited historical data available. Some features may be restricted.")
            hist = pd.DataFrame()

        # ==========================================
        # QUANT STACK ANALYSIS (Sniper Enhancement)
        # ==========================================
        st.title(f"🎯 Sniper Intelligence: {ticker}")
        st.success(f"**{ticker}** | Spot: `${spot:.2f}` | Source: `{src}`")
        
        st.markdown("### 🏛️ Quant Stack Analysis")
        
        try:
            qs = QuantStackEngine(ticker, hist)
            
            # Layer 1: Macro Regime
            with st.spinner("Analyzing Macro Regime..."):
                regime, macro_score, macro_df = qs.run_layer1_regime()
            
            # Layer 2: Kalman Filter
            kalman_df = qs.run_layer2_kalman()
            
            # Layer 3: GARCH Volatility
            sizing, vol_ratio, garch_status = qs.run_layer3_garch()
            
            # --- DISPLAY THE 3 LAYERS ---
            c1, c2, c3 = st.columns(3)
            
            # 1. Macro Direction Card
            with c1:
                color = "#10B981" if regime == "BULL" else "#EF4444" if regime == "BEAR" else "#64748B"
                st.markdown(f"""
                <div style="border:1px solid #E2E8F0; padding:15px; border-radius:8px; border-top: 4px solid {color};">
                    <h4 style="margin:0; color:#64748B;">L1: DIRECTION</h4>
                    <h2 style="margin:5px 0; color:{color};">{regime}</h2>
                    <small>Macro Score: {macro_score:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
                
            # 2. Kalman Valuation Card
            with c2:
                if kalman_df is not None and not kalman_df.empty:
                    kalman_z = kalman_df['Kalman_Z'].iloc[-1]
                    if kalman_z > 2.0: k_status, k_color = "OVERBOUGHT", "#EF4444"
                    elif kalman_z < -2.0: k_status, k_color = "OVERSOLD", "#10B981"
                    else: k_status, k_color = "FAIR VALUE", "#3B82F6"
                    
                    st.markdown(f"""
                    <div style="border:1px solid #E2E8F0; padding:15px; border-radius:8px; border-top: 4px solid {k_color};">
                        <h4 style="margin:0; color:#64748B;">L2: VALUATION</h4>
                        <h2 style="margin:5px 0; color:{k_color};">{k_status}</h2>
                        <small>Kalman Z: {kalman_z:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    kalman_z = 0
                    st.info("Insufficient data for Kalman")
                
            # 3. Sizing Card
            with c3:
                s_color = "#10B981" if sizing >= 1.0 else "#F59E0B" if sizing >= 0.5 else "#EF4444"
                st.markdown(f"""
                <div style="border:1px solid #E2E8F0; padding:15px; border-radius:8px; border-top: 4px solid {s_color};">
                    <h4 style="margin:0; color:#64748B;">L3: SIZING</h4>
                    <h2 style="margin:5px 0; color:{s_color};">{sizing:.1f}x</h2>
                    <small>{garch_status}</small>
                </div>
                """, unsafe_allow_html=True)

            # --- KALMAN VISUALIZATION ---
            if kalman_df is not None and not kalman_df.empty:
                st.markdown("#### 🔭 Kalman Trajectory (True Value vs Noise)")
                
                # Calculate VWAP
                if 'Volume' in kalman_df.columns and 'High' in kalman_df.columns and 'Low' in kalman_df.columns:
                    kalman_df['TP'] = (kalman_df['High'] + kalman_df['Low'] + kalman_df['Close']) / 3
                    kalman_df['TPV'] = kalman_df['TP'] * kalman_df['Volume']
                    kalman_df['VWAP'] = kalman_df['TPV'].cumsum() / kalman_df['Volume'].cumsum()
                    has_vwap = True
                else:
                    has_vwap = False
                
                fig_k = go.Figure()
                fig_k.add_trace(go.Scatter(x=kalman_df.index, y=kalman_df['Close'], name='Price', line=dict(color='gray', width=1)))
                fig_k.add_trace(go.Scatter(x=kalman_df.index, y=kalman_df['Kalman'], name='Kalman True Value', line=dict(color='#3B82F6', width=2)))
                
                if has_vwap:
                    fig_k.add_trace(go.Scatter(x=kalman_df.index, y=kalman_df['VWAP'], name='VWAP', line=dict(color='#F59E0B', width=2, dash='dash')))
                
                upper = kalman_df['Kalman'] + (2 * kalman_df['Res_Std'])
                lower = kalman_df['Kalman'] - (2 * kalman_df['Res_Std'])
                fig_k.add_trace(go.Scatter(x=kalman_df.index, y=upper, name='+2σ Band', mode='lines', line=dict(width=0), showlegend=False))
                fig_k.add_trace(go.Scatter(x=kalman_df.index, y=lower, name='-2σ Band', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)'))
                
                fig_k.update_layout(template='plotly_white', height=400, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_k, use_container_width=True)
                
                # VWAP Status
                if has_vwap:
                    current_price = kalman_df['Close'].iloc[-1]
                    current_vwap = kalman_df['VWAP'].iloc[-1]
                    vwap_delta = ((current_price - current_vwap) / current_vwap) * 100
                    vwap_direction = "ABOVE" if current_price > current_vwap else "BELOW"
                    vwap_color = "#10B981" if current_price > current_vwap else "#EF4444"
                    
                    st.markdown(f"""
                    <div style="border:1px solid #E2E8F0; padding:10px; border-radius:6px; background:#F8FAFC; margin-top:10px;">
                        <strong>VWAP:</strong> <span style="color:{vwap_color}; font-weight:600;">${current_vwap:.2f}</span> | 
                        Price is <span style="color:{vwap_color}; font-weight:600;">{vwap_direction}</span> by {abs(vwap_delta):.2f}%
                    </div>
                    """, unsafe_allow_html=True)

            # --- FINAL DECISION MATRIX ---
            st.markdown("#### 🤖 Final Decision Matrix")
            
            decision = "WAIT / CASH"
            d_desc = "Conditions not met."
            d_class = "signal-box"
            
            if regime == "BULL":
                if kalman_z < 2.0:
                    decision = "✅ ENTER / LONG"
                    d_desc = f"Bullish Macro + Fair Valuation. Suggested Size: {sizing:.1f}x"
                    d_class = "signal-buy"
                else:
                    decision = "⛔ TRAPPED (WAIT)"
                    d_desc = "Bullish Macro but Price is statistically overextended (Z > 2.0)."
                    d_class = "signal-sell"
            elif regime == "BEAR":
                decision = "🛡️ DEFENSE / CASH"
                d_desc = "Bearish Macro Regime. Do not fight the Four Horsemen."
                d_class = "signal-sell"
            else:
                decision = "⚖️ NEUTRAL / WAIT"
                d_desc = "Macro conditions are mixed. Exercise caution."
                d_class = "signal-box"
            
            st.markdown(f"""
            <div class="{d_class}">
                <h3>STRATEGY COMMAND</h3>
                <h1>{decision}</h1>
                <p>{d_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Quant Stack calculation error: {e}")
        
        st.markdown("---")

    # ==========================================
    # MODULE A: DASHBOARD (Market Command Center)
    # ==========================================
    if mode == "📺 Dashboard":
        st.title("📺 Market Command Center")
        st.caption("Live indices, market sentiment, and breaking news")

        # Fetch all major indices
        @st.cache_data(ttl=120, show_spinner=False)
        def fetch_indices():
            indices = {
                "SPY": "S&P 500",
                "QQQ": "NASDAQ 100", 
                "DIA": "DOW 30",
                "IWM": "Russell 2000",
                "^VIX": "VIX (Fear)"
            }
            data = {}
            last_date = None
            for sym, name in indices.items():
                try:
                    df = yf.download(sym, period="5d", progress=False)
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df = df.droplevel(1, axis=1)
                        current = float(df["Close"].iloc[-1])
                        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else current
                        change = ((current - prev) / prev) * 100
                        data[name] = {"price": current, "change": change, "symbol": sym}
                        if last_date is None:
                            last_date = df.index[-1]
                except:
                    pass
            return data, last_date

        @st.cache_data(ttl=300, show_spinner=False)
        def fetch_news():
            """Fetch news from multiple high-quality RSS feeds with improved sentiment analysis."""
            import re
            news_items = []
            
            # Diverse RSS feeds for better coverage
            rss_urls = [
                ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC"),
                ("https://www.marketwatch.com/rss/marketpulse", "MarketWatch"),
                ("https://finance.yahoo.com/rss/topstories", "Yahoo"),
                ("https://news.google.com/rss/search?q=stock+market+when:1d&hl=en-US&gl=US&ceid=US:en", "Google"),
            ]
            
            # Enhanced sentiment keywords
            positive = ["surge", "jump", "rally", "gain", "gains", "rise", "rises", "soar", "beat", "beats", 
                       "record", "high", "highs", "bull", "bullish", "growth", "profit", "profits", 
                       "buy", "upgrade", "boom", "strength", "strong", "outperform", "breakout"]
            negative = ["fall", "falls", "drop", "drops", "crash", "plunge", "plunges", "sink", "sinks",
                       "miss", "misses", "low", "lows", "bear", "bearish", "loss", "losses", "fear", 
                       "sell", "cut", "cuts", "down", "decline", "declines", "warn", "warning", "slump", 
                       "tumble", "tumbles", "weak", "weakness", "underperform"]
            
            for url, source in rss_urls:
                try:
                    response = requests.get(url, timeout=5, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    })
                    if response.status_code == 200:
                        content = response.text
                        items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                        
                        for item_xml in items[:4]:  # Limit per source
                            # Extract title
                            title_match = re.search(r'<title><!\\[CDATA\\[(.*?)\\]\\]></title>|<title>(.*?)</title>', item_xml)
                            link_match = re.search(r'<link>(.*?)</link>', item_xml)
                            
                            title = ""
                            if title_match:
                                title = title_match.group(1) or title_match.group(2) or ""
                            
                            # Clean HTML entities
                            title = re.sub(r'<[^>]+>', '', title).strip()
                            link = link_match.group(1) if link_match else f"https://{source.lower()}.com"
                            
                            if not title or len(title) < 10:
                                continue
                            
                            # Calculate sentiment score
                            title_lower = title.lower()
                            sentiment = 0
                            for word in positive:
                                if word in title_lower:
                                    sentiment += 1
                            for word in negative:
                                if word in title_lower:
                                    sentiment -= 1
                            
                            news_items.append({
                                "title": title[:200],  # Truncate long titles
                                "publisher": source,
                                "link": link.strip(),
                                "time": len(news_items),
                                "sentiment": sentiment,
                                "ticker": source
                            })
                except Exception:
                    continue
            
            # Fallback message if no news
            if not news_items:
                news_items = [{
                    "title": "Markets are active. Check financial news sites for latest updates.",
                    "publisher": "Marta Tools",
                    "link": "https://finance.yahoo.com",
                    "time": 0,
                    "sentiment": 0,
                    "ticker": "INFO"
                }]
            
            return news_items[:15]  # Return top 15

        # Refresh button
        col_refresh, col_date = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh", help="Clear cache and reload data"):
                st.cache_data.clear()
                st.rerun()
        
        with st.spinner("Loading market data..."):
            indices_data, data_date = fetch_indices()
            news_data = fetch_news()
        
        # Show data freshness
        with col_date:
            if data_date:
                date_str = data_date.strftime('%b %d, %Y') if hasattr(data_date, 'strftime') else str(data_date)
                st.caption(f"📅 Data as of: {date_str}")

        # --- INDICES TICKER BAR ---
        if indices_data:
            cols = st.columns(len(indices_data))
            for i, (name, data) in enumerate(indices_data.items()):
                with cols[i]:
                    if "VIX" in name:
                        # VIX: up is bad (red), down is good (green) - inverse logic
                        st.metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f}%", delta_color="inverse")
                    else:
                        # Normal indices: up is good (green), down is bad (red)
                        st.metric(name, f"${data['price']:.2f}", f"{data['change']:+.2f}%", delta_color="normal")
        else:
            st.warning("Unable to load indices data")

        # --- SENTIMENT + CHART ROW ---
        st.markdown("---")
        col_sent, col_chart = st.columns([1, 2])
        
        with col_sent:
            if indices_data:
                # Calculate market breadth
                bullish_count = sum(1 for d in indices_data.values() if d.get("change", 0) > 0 and "VIX" not in d.get("symbol", ""))
                total_indices = len([d for d in indices_data.values() if "VIX" not in d.get("symbol", "")])
                breadth_ratio = bullish_count / total_indices if total_indices > 0 else 0.5
                
                # Get VIX level
                vix_data = indices_data.get("VIX (Fear)", {})
                vix_level = vix_data.get("price", 20)
                
                # Calculate news sentiment
                if news_data:
                    avg_news_sentiment = sum(n["sentiment"] for n in news_data) / len(news_data) if news_data else 0
                else:
                    avg_news_sentiment = 0
                
                # Composite Fear/Greed Score (0-100)
                # VIX: inverted and normalized (low VIX = greed, high VIX = fear)
                vix_score = max(0, min(100, 100 - (vix_level - 10) * 5))  # VIX 10=100, 30=0
                
                # Breadth: percentage of indices up
                breadth_score = breadth_ratio * 100
                
                # News: normalized to 0-100
                news_score = max(0, min(100, 50 + (avg_news_sentiment * 10)))
                
                # Weighted composite (VIX 40%, Breadth 30%, News 30%)
                composite_score = (vix_score * 0.4) + (breadth_score * 0.3) + (news_score * 0.3)
                
                # Classify sentiment
                if composite_score >= 70:
                    fear_level, fear_emoji, fear_color = "EXTREME GREED", "🤑", "#10B981"
                elif composite_score >= 55:
                    fear_level, fear_emoji, fear_color = "GREED", "😊", "#10B981"
                elif composite_score >= 45:
                    fear_level, fear_emoji, fear_color = "NEUTRAL", "😐", "#F59E0B"
                elif composite_score >= 30:
                    fear_level, fear_emoji, fear_color = "FEAR", "😰", "#EF4444"
                else:
                    fear_level, fear_emoji, fear_color = "EXTREME FEAR", "😱", "#EF4444"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {fear_color}15, {fear_color}30); 
                            padding: 24px; border-radius: 16px; border: 1px solid {fear_color}40; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">{fear_emoji}</div>
                    <h2 style="margin: 0; color: {fear_color}; font-size: 1.8rem; letter-spacing: 2px;">{fear_level}</h2>
                    <p style="margin: 12px 0 0 0; color: #64748B; font-size: 0.9rem;">
                        Score: <strong>{composite_score:.0f}/100</strong> · VIX: <strong>{vix_level:.1f}</strong><br/>
                        {bullish_count}/{total_indices} indices up
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
                
                # News sentiment card
                if news_data:
                    avg_sentiment = sum(n["sentiment"] for n in news_data) / len(news_data) if news_data else 0
                    if avg_sentiment > 0.3:
                        sent_label, sent_emoji, sent_color = "Bullish", "📈", "#10B981"
                    elif avg_sentiment < -0.3:
                        sent_label, sent_emoji, sent_color = "Bearish", "📉", "#EF4444"
                    else:
                        sent_label, sent_emoji, sent_color = "Neutral", "➡️", "#64748B"
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0;">
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;">News Sentiment</div>
                        <div style="font-size: 1.4rem; font-weight: 600; color: {sent_color}; margin-top: 4px;">
                            {sent_emoji} {sent_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("<p style='font-size: 0.85rem; color: #64748B; margin-bottom: 8px;'>S&P 500 · 5 Day</p>", unsafe_allow_html=True)
            try:
                spy_hist = yf.download("SPY", period="5d", interval="15m", progress=False)
                if spy_hist is not None and not spy_hist.empty:
                    if isinstance(spy_hist.columns, pd.MultiIndex):
                        spy_hist = spy_hist.droplevel(1, axis=1)
                    
                    fig_spy = go.Figure()
                    fig_spy.add_trace(go.Scatter(
                        x=spy_hist.index, 
                        y=spy_hist["Close"],
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#10B981" if spy_hist["Close"].iloc[-1] > spy_hist["Close"].iloc[0] else "#EF4444", width=2),
                        fillcolor="rgba(16, 185, 129, 0.1)" if spy_hist["Close"].iloc[-1] > spy_hist["Close"].iloc[0] else "rgba(239, 68, 68, 0.1)"
                    ))
                    fig_spy.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        template="plotly_white",
                        showlegend=False,
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#E2E8F0")
                    )
                    st.plotly_chart(fig_spy, use_container_width=True)
            except:
                st.info("Chart loading...")

        # --- NEWS FEED ---
        st.markdown("---")
        st.markdown("### 📰 Breaking News & Headlines")
        
        if news_data and len(news_data) > 0:
            for news in news_data[:10]:
                title = news.get("title", "")
                if not title or len(title) < 5:
                    continue
                    
                publisher = news.get("publisher", "News")
                ticker = news.get("ticker", "")
                link = news.get("link", "#")
                sentiment = news.get("sentiment", 0)
                
                sent_icon = "🟢" if sentiment > 0 else "🔴" if sentiment < 0 else "⚪"
                sent_bg = "rgba(16, 185, 129, 0.1)" if sentiment > 0 else "rgba(239, 68, 68, 0.1)" if sentiment < 0 else "transparent"
                
                st.markdown(f"""
                <div style="padding: 12px; margin: 8px 0; border-radius: 8px; background: {sent_bg}; border-left: 3px solid {'#10B981' if sentiment > 0 else '#EF4444' if sentiment < 0 else '#94A3B8'};">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">{publisher} • {ticker}</span>
                            <div style="margin-top: 4px;">
                                <a href="{link}" target="_blank" style="text-decoration: none; color: #0F172A; font-weight: 500; font-size: 0.95rem;">
                                    {title}
                                </a>
                            </div>
                        </div>
                        <span style="font-size: 1.2rem; margin-left: 10px;">{sent_icon}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📡 News feed is loading... If this persists, the news API may be temporarily unavailable.")

        # --- SECTOR HEATMAP (Quick View) ---
        st.markdown("---")
        st.markdown("### 🏢 Sector Pulse")
        
        @st.cache_data(ttl=300, show_spinner=False)
        def fetch_sectors():
            sectors = {
                "XLK": "Tech", "XLF": "Finance", "XLE": "Energy", 
                "XLV": "Health", "XLY": "Consumer", "XLI": "Industrial"
            }
            results = {}
            for sym, name in sectors.items():
                try:
                    df = yf.download(sym, period="2d", progress=False)
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df = df.droplevel(1, axis=1)
                        curr = float(df["Close"].iloc[-1])
                        prev = float(df["Close"].iloc[0])
                        chg = ((curr - prev) / prev) * 100
                        results[name] = chg
                except:
                    pass
            return results

        sectors = fetch_sectors()
        if sectors:
            sector_cols = st.columns(len(sectors))
            for i, (name, change) in enumerate(sectors.items()):
                with sector_cols[i]:
                    color = "#10B981" if change >= 0 else "#EF4444"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background: {color}22; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #64748B;">{name}</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{change:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ==========================================
    # MODULE B: OPTIONIZER
    # ==========================================
    elif mode == "💎 Options":
        st.title("💎 Optionizer")
        st.caption("Options pricing engine with Greeks, edge detection, and backtesting")

        # Config inputs at top of Optionizer
        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        with cfg1:
            ticker = st.text_input("TICKER", value="TSLA").upper()
        with cfg2:
            rf = st.number_input("RISK FREE (%)", value=4.5)
        with cfg3:
            cap = st.number_input("CAPITAL ($)", value=100000)
        with cfg4:
            kelly = st.slider("KELLY FACTOR", 0.1, 1.0, 0.5)

        st.markdown("---")

        hydra = HydraEngine()
        with st.spinner(f"Fetching data for {ticker}..."):
            spot, src = hydra.get_spot(ticker)
            hist = hydra.get_history(ticker)

        if spot is None:
            st.error(f"❌ Unable to retrieve market data for **{ticker}**. Please check the ticker symbol or try again.")
            with st.expander("🔧 Debug Info"):
                st.code(hydra.get_debug_info())
                st.info("Try running: `pip install --upgrade yfinance` in your terminal")
            st.stop()

        st.success(f"**{ticker}** | Spot: `${spot:.2f}` | Source: `{src}`")

        with st.spinner(f"Fetching expirations for {ticker}..."):
            exps = hydra.get_expirations(ticker)

        if not exps:
            st.error("No expirations available for this ticker.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_exp = st.selectbox("EXPIRATION", exps)

        chain, _ = hydra.get_chain(ticker, sel_exp, spot)

        if chain is None or chain.empty:
            st.error("No option chain available for this expiration.")
            return

        surface = SurfaceEngine(chain)

        # Precompute fair value and edge across chain to suggest optimal trades
        dt_full = (datetime.strptime(sel_exp, "%Y-%m-%d").date() - date.today()).days
        T_full = max(dt_full / 365.0, 0.001)

        def model_fair(row):
            sigma_row = surface.get_iv(row["strike"]) if surface.valid else row["impliedVolatility"]
            sigma_row = max(sigma_row if sigma_row else row["impliedVolatility"], 1e-4)
            engine = PricingKernel(spot, row["strike"], T_full, rf / 100, sigma_row, row["type"])
            bs_p = engine.price_bsm()
            hes_p = engine.price_heston_proxy()
            mc_p = engine.price_mc(sims=20000)
            return (bs_p * 0.4) + (hes_p * 0.3) + (mc_p * 0.3)

        chain = chain.copy()
        chain["fair"] = chain.apply(model_fair, axis=1)
        chain["edge"] = chain["fair"] - chain["mid"]

        top_calls = chain[chain["type"] == "call"].nlargest(3, "edge")
        top_puts = chain[chain["type"] == "put"].nlargest(3, "edge")

        st.markdown("### 🏅 Optimal Edge Picks")
        col_tc, col_tp = st.columns(2)
        with col_tc:
            st.markdown("**Top Calls (by edge)**")
            if not top_calls.empty:
                st.dataframe(
                    top_calls[["contractSymbol", "strike", "mid", "fair", "edge", "impliedVolatility", "volume", "openInterest"]]
                    .rename(columns={"mid": "Market", "fair": "Model", "edge": "Edge", "impliedVolatility": "IV"})
                    .style.format({"Market": "${:.2f}", "Model": "${:.2f}", "Edge": "${:.2f}", "IV": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No call candidates available.")
        with col_tp:
            st.markdown("**Top Puts (by edge)**")
            if not top_puts.empty:
                st.dataframe(
                    top_puts[["contractSymbol", "strike", "mid", "fair", "edge", "impliedVolatility", "volume", "openInterest"]]
                    .rename(columns={"mid": "Market", "fair": "Model", "edge": "Edge", "impliedVolatility": "IV"})
                    .style.format({"Market": "${:.2f}", "Model": "${:.2f}", "Edge": "${:.2f}", "IV": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No put candidates available.")

        with c2:
            strikes = sorted(chain["strike"].unique())
            sel_strike = st.selectbox("STRIKE", strikes, index=len(strikes) // 2 if strikes else 0)
        with c3:
            sel_type = st.radio("TYPE", ["CALL", "PUT"], horizontal=True)

        if st.button("RUN TRINITY ENGINE", type="primary"):
            try:
                row = chain[(chain["strike"] == sel_strike) & (chain["type"] == sel_type.lower())].iloc[0]
                mkt_p, mkt_iv = row["mid"], row["impliedVolatility"]

                sigma = surface.get_iv(sel_strike) if surface.valid else None
                if sigma is None or sigma <= 0:
                    sigma = mkt_iv

                # Calculate IV Rank (52-week)
                @st.cache_data(ttl=600, show_spinner=False)
                def calc_iv_rank(ticker_sym, current_iv):
                    try:
                        # Get 1 year of historical data
                        hist_data = yf.download(ticker_sym, period="1y", progress=False)
                        if hist_data is None or hist_data.empty:
                            return None, None, None
                        
                        if isinstance(hist_data.columns, pd.MultiIndex):
                            hist_data = hist_data.droplevel(1, axis=1)
                        
                        # Calculate rolling 20-day realized volatility as IV proxy
                        returns = np.log(hist_data["Close"] / hist_data["Close"].shift(1))
                        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                        rolling_vol = rolling_vol.dropna()
                        
                        if len(rolling_vol) < 20:
                            return None, None, None
                        
                        iv_low = rolling_vol.min()
                        iv_high = rolling_vol.max()
                        
                        # IV Rank = where current IV sits in 52-week range
                        if iv_high > iv_low:
                            iv_rank = (current_iv - iv_low) / (iv_high - iv_low)
                            iv_rank = float(np.clip(iv_rank, 0, 1))
                        else:
                            iv_rank = 0.5
                        
                        return iv_rank, float(iv_low), float(iv_high)
                    except Exception:
                        return None, None, None

                iv_rank, iv_52w_low, iv_52w_high = calc_iv_rank(ticker, mkt_iv)

                dt = (datetime.strptime(sel_exp, "%Y-%m-%d").date() - date.today()).days
                T = max(dt / 365.0, 0.001)

                engine = PricingKernel(spot, sel_strike, T, rf / 100, sigma, sel_type)
                bs = engine.price_bsm()
                hes = engine.price_heston_proxy()
                mc = engine.price_mc(sims=50000)
                greeks = engine.get_greeks()

                consensus = (bs * 0.4) + (hes * 0.3) + (mc * 0.3)
                edge = consensus - mkt_p

                # Probability of Profit (break-even based, risk-neutral) - calculate first for Kelly
                breakeven = sel_strike + mkt_p if sel_type == "CALL" else sel_strike - mkt_p
                drift = (rf / 100 - 0.5 * sigma**2) * T
                denom = sigma * np.sqrt(T)
                if denom > 0:
                    if sel_type == "CALL":
                        pop = 1 - stats.norm.cdf((np.log(breakeven / spot) - drift) / denom)
                    else:
                        pop = stats.norm.cdf((np.log(breakeven / spot) - drift) / denom)
                    pop = float(np.clip(pop, 0, 1))
                else:
                    pop = 0.5  # Default to 50% if can't calculate

                # Kelly Criterion Position Sizing
                # Formula: Kelly% = (edge / premium) * adjustment * user_factor
                # Only allocate if we have positive edge
                if edge > 0 and mkt_p > 0:
                    # Method 1: Edge-based Kelly (edge relative to cost)
                    edge_ratio = edge / mkt_p
                    
                    # Method 2: PoP-based Kelly (2*p - 1 for fair odds)
                    pop_kelly = max(0, 2 * pop - 1) if not np.isnan(pop) else 0
                    
                    # Blend both methods and apply user's kelly factor
                    raw_kelly = (edge_ratio * 0.5 + pop_kelly * 0.5)
                    k_alloc = min(raw_kelly * kelly, 0.25)  # Cap at 25% max position
                    k_alloc = max(k_alloc, 0)
                else:
                    k_alloc = 0

                # Scenario outcomes (±10% and spot)
                scenarios = [0.9 * spot, spot, 1.1 * spot]
                payoffs = []
                for s in scenarios:
                    intrinsic = max(s - sel_strike, 0) if sel_type == "CALL" else max(sel_strike - s, 0)
                    payoffs.append(intrinsic - mkt_p)

                st.markdown("---")
                c_res1, c_res2 = st.columns([1, 2])

                with c_res1:
                    cls = "signal-buy" if edge > 0 else "signal-sell"
                    sig = "LONG" if edge > 0 else "SHORT"
                    
                    # IV Rank color coding
                    if iv_rank is not None:
                        if iv_rank > 0.7:
                            iv_color = "#EF4444"  # Red - high IV (good for selling)
                            iv_label = "HIGH"
                        elif iv_rank < 0.3:
                            iv_color = "#10B981"  # Green - low IV (good for buying)
                            iv_label = "LOW"
                        else:
                            iv_color = "#F59E0B"  # Yellow - mid IV
                            iv_label = "MID"
                        iv_display = f"{iv_rank:.0%}"
                    else:
                        iv_color = "#64748B"
                        iv_label = "N/A"
                        iv_display = "N/A"
                    
                    st.markdown(
                        f"""
                    <div class="signal-box {cls}">
                        <h3>{sig} SIGNAL</h3>
                        <h1>${edge:+.2f} EDGE</h1>
                        <hr style="border-color:#E2E8F0">
                        <small>KELLY SIZE: {k_alloc:.1%}</small><br>
                        <strong>${cap * k_alloc:.2f}</strong>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    
                    # IV Rank Display
                    if iv_rank is not None and iv_52w_low is not None and iv_52w_high is not None:
                        st.markdown(
                            f"""
                        <div style="background: linear-gradient(135deg, {iv_color}15, {iv_color}30); 
                                    padding: 16px; border-radius: 12px; border: 1px solid {iv_color}40; margin-top: 12px;">
                            <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">IV Rank (52-Week)</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: {iv_color};">{iv_display} <span style="font-size: 0.9rem;">{iv_label}</span></div>
                            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 4px;">
                                Range: {iv_52w_low:.1%} - {iv_52w_high:.1%} | Current: {mkt_iv:.1%}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("IV Rank data unavailable")

                with c_res2:
                    df = pd.DataFrame(
                        {
                            "Model": ["Market", "Black-Scholes", "Heston", "Monte Carlo"],
                            "Price": [mkt_p, bs, hes, mc],
                        }
                    )
                    st.dataframe(df.style.format({"Price": "${:.2f}"}), use_container_width=True)

                st.subheader("📐 3rd Order Greeks")
                g1, g2, g3, g4, g5, g6 = st.columns(6)
                g1.metric("Delta", f"{greeks.delta:.3f}")
                g2.metric("Gamma", f"{greeks.gamma:.3f}")
                g3.metric("Vega", f"{greeks.vega:.3f}")
                g4.metric("Vanna", f"{greeks.vanna:.3f}")
                g5.metric("Volga", f"{greeks.volga:.3f}")
                g6.metric("Speed", f"{greeks.speed:.3f}")

                st.subheader("📊 Outcome & Probability")
                o1, o2, o3 = st.columns(3)
                o1.metric("Prob. of Profit", f"{pop:.1%}" if not np.isnan(pop) else "N/A")
                o2.metric("Breakeven", f"${breakeven:.2f}")
                o3.metric("Model Edge", f"${edge:+.2f}")

                scen_df = pd.DataFrame(
                    {
                        "Underlying": [f"{s:.2f}" for s in scenarios],
                        "P/L": [f"${p:.2f}" for p in payoffs],
                    }
                )
                st.dataframe(scen_df, use_container_width=True, hide_index=True)

                # Historical PoP backtest (realized moves vs current premium)
                def backtest_pop(close_series, premium, strike, option_type, horizon_days):
                    if close_series is None or close_series.empty or horizon_days <= 1:
                        return None
                    closes = close_series.dropna()
                    if len(closes) <= horizon_days:
                        return None
                    wins, total = 0, 0
                    pnls = []
                    for i in range(len(closes) - horizon_days):
                        spot_i = closes.iloc[i]
                        future = closes.iloc[i + horizon_days]
                        if option_type == "CALL":
                            intrinsic = max(future - strike, 0)
                        else:
                            intrinsic = max(strike - future, 0)
                        pnl = intrinsic - premium
                        pnls.append(pnl)
                        if pnl > 0:
                            wins += 1
                        total += 1
                    if total == 0:
                        return None
                    return {
                        "pop_hist": wins / total,
                        "avg_pnl": np.mean(pnls),
                        "med_pnl": np.median(pnls),
                        "samples": total,
                    }

                horizon_days = max(int(T * 365), 1)
                bt_res = backtest_pop(hist["Close"], mkt_p, sel_strike, sel_type, horizon_days)
                if bt_res:
                    st.subheader("🧪 Historical Check (Realized Moves vs Premium)")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Hist Win Rate", f"{bt_res['pop_hist']:.1%}")
                    b2.metric("Avg P/L", f"${bt_res['avg_pnl']:.2f}")
                    b3.metric("Median P/L", f"${bt_res['med_pnl']:.2f}")
                    b4.metric("Samples", f"{bt_res['samples']}")
                else:
                    st.info("Not enough historical samples to compute a realized PoP backtest for this tenor.")

                # Rolling historical walk-through using ATM options with realized vol as proxy
                def walk_backtest(close_series, tenor_days, r_rate, option_type):
                    closes = close_series.dropna()
                    if len(closes) < tenor_days + 40:
                        return None
                    entries = []
                    for i in range(40, len(closes) - tenor_days):
                        window = closes.iloc[i - 40 : i]
                        if window.isnull().any():
                            continue
                        sigma_i = window.pct_change().std() * np.sqrt(252)
                        if sigma_i is None or sigma_i <= 0:
                            continue
                        spot_i = closes.iloc[i]
                        strike_i = spot_i  # ATM
                        T_i = tenor_days / 365.0
                        engine_i = PricingKernel(spot_i, strike_i, T_i, r_rate / 100, sigma_i, option_type)
                        price_i = engine_i.price_bsm()

                        # Model PoP using breakeven
                        breakeven_i = strike_i + price_i if option_type == "CALL" else strike_i - price_i
                        drift_i = (r_rate / 100 - 0.5 * sigma_i**2) * T_i
                        denom_i = sigma_i * np.sqrt(T_i)
                        if denom_i > 0:
                            if option_type == "CALL":
                                pop_i = 1 - stats.norm.cdf((np.log(breakeven_i / spot_i) - drift_i) / denom_i)
                            else:
                                pop_i = stats.norm.cdf((np.log(breakeven_i / spot_i) - drift_i) / denom_i)
                            pop_i = float(np.clip(pop_i, 0, 1))
                        else:
                            pop_i = np.nan

                        future = closes.iloc[i + tenor_days]
                        intrinsic = max(future - strike_i, 0) if option_type == "CALL" else max(strike_i - future, 0)
                        pnl_i = intrinsic - price_i
                        entries.append(
                            {
                                "entry_date": closes.index[i],
                                "expiry_date": closes.index[i + tenor_days],
                                "spot": spot_i,
                                "price": price_i,
                                "breakeven": breakeven_i,
                                "pop_model": pop_i,
                                "pnl": pnl_i,
                                "win": 1 if pnl_i > 0 else 0,
                            }
                        )
                    return pd.DataFrame(entries) if entries else None

                walk_df = walk_backtest(hist["Close"], horizon_days, rf, sel_type)
                if walk_df is not None and not walk_df.empty:
                    st.subheader("📈 Rolling Backtest (ATM, same tenor)")
                    win_rate = walk_df["win"].mean()
                    avg_pnl = walk_df["pnl"].mean()
                    med_pnl = walk_df["pnl"].median()
                    avg_pop = walk_df["pop_model"].mean()
                    ww1, ww2, ww3, ww4 = st.columns(4)
                    ww1.metric("Realized Win Rate", f"{win_rate:.1%}")
                    ww2.metric("Avg P/L", f"${avg_pnl:.2f}")
                    ww3.metric("Median P/L", f"${med_pnl:.2f}")
                    ww4.metric("Avg Model PoP", f"{avg_pop:.1%}")

                    walk_df["cum_win_rate"] = walk_df["win"].expanding().mean()
                    walk_df["pop_model_smooth"] = walk_df["pop_model"].rolling(5).mean()

                    fig_bt = go.Figure()
                    fig_bt.add_trace(
                        go.Scatter(
                            x=walk_df["expiry_date"],
                            y=walk_df["cum_win_rate"],
                            mode="lines",
                            name="Realized Win Rate (cum)",
                            line=dict(color="#10B981", width=2),
                        )
                    )
                    fig_bt.add_trace(
                        go.Scatter(
                            x=walk_df["expiry_date"],
                            y=walk_df["pop_model_smooth"],
                            mode="lines",
                            name="Model PoP (5-period avg)",
                            line=dict(color="#3B82F6", width=2, dash="dash"),
                        )
                    )
                    fig_bt.update_layout(height=320, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_bt, use_container_width=True)

                    with st.expander("Recent Backtest Trades (last 10)"):
                        st.dataframe(
                            walk_df.tail(10)[["entry_date", "expiry_date", "spot", "price", "breakeven", "pop_model", "pnl", "win"]]
                            .rename(
                                columns={
                                    "spot": "Spot",
                                    "price": "Premium",
                                    "breakeven": "Breakeven",
                                    "pop_model": "Model_PoP",
                                    "pnl": "PnL",
                                    "win": "Win",
                                }
                            )
                            .style.format(
                                {
                                    "Spot": "${:.2f}",
                                    "Premium": "${:.2f}",
                                    "Breakeven": "${:.2f}",
                                    "Model_PoP": "{:.1%}",
                                    "PnL": "${:.2f}",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.info("Backtest: not enough history to run rolling ATM simulation for this tenor.")

            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.subheader("📈 Volatility Skew & Mispricing Heatmap")
        
        # Enhanced skew chart with mispricing visualization
        fig_s = go.Figure()
        
        # Prepare data for visualization
        calls_data = chain[chain["type"] == "call"].copy()
        puts_data = chain[chain["type"] == "put"].copy()
        
        # Calculate edge for color coding
        if "edge" in chain.columns:
            # Normalize edge for color intensity
            max_edge = max(abs(chain["edge"].max()), abs(chain["edge"].min()), 0.01)
            
            # Add calls with edge coloring
            if not calls_data.empty:
                calls_colors = ['rgb(16, 185, 129)' if e > 0 else 'rgb(239, 68, 68)' for e in calls_data["edge"]]
                calls_sizes = [8 + abs(e) / max_edge * 20 for e in calls_data["edge"]]
                
                fig_s.add_trace(go.Scatter(
                    x=calls_data["strike"],
                    y=calls_data["impliedVolatility"],
                    mode="markers",
                    name="Calls",
                    marker=dict(
                        size=calls_sizes,
                        color=calls_data["edge"],
                        colorscale=[[0, 'rgb(239, 68, 68)'], [0.5, 'rgb(148, 163, 184)'], [1, 'rgb(16, 185, 129)']],
                        cmin=-max_edge,
                        cmax=max_edge,
                        showscale=True,
                        colorbar=dict(title="Edge $", x=1.02, len=0.5, y=0.75),
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Strike: ${s:.0f}<br>IV: {iv:.1%}<br>Edge: ${e:.2f}<br>Vol: {v:,.0f}" 
                          for s, iv, e, v in zip(calls_data["strike"], calls_data["impliedVolatility"], 
                                                  calls_data["edge"], calls_data["volume"])],
                    hoverinfo="text"
                ))
            
            # Add puts with edge coloring
            if not puts_data.empty:
                puts_sizes = [8 + abs(e) / max_edge * 20 for e in puts_data["edge"]]
                
                fig_s.add_trace(go.Scatter(
                    x=puts_data["strike"],
                    y=puts_data["impliedVolatility"],
                    mode="markers",
                    name="Puts",
                    marker=dict(
                        size=puts_sizes,
                        color=puts_data["edge"],
                        colorscale=[[0, 'rgb(239, 68, 68)'], [0.5, 'rgb(148, 163, 184)'], [1, 'rgb(16, 185, 129)']],
                        cmin=-max_edge,
                        cmax=max_edge,
                        symbol="diamond",
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Strike: ${s:.0f}<br>IV: {iv:.1%}<br>Edge: ${e:.2f}<br>Vol: {v:,.0f}" 
                          for s, iv, e, v in zip(puts_data["strike"], puts_data["impliedVolatility"], 
                                                  puts_data["edge"], puts_data["volume"])],
                    hoverinfo="text"
                ))
        else:
            # Fallback if no edge calculated
            fig_s.add_trace(go.Scatter(
                x=chain["strike"], 
                y=chain["impliedVolatility"], 
                mode="markers", 
                name="Market IV",
                marker=dict(size=10, color="#3B82F6")
            ))
        
        # Add spline fit
        if surface.valid:
            xs = np.linspace(chain["strike"].min(), chain["strike"].max(), 100)
            ys = [surface.get_iv(x) for x in xs]
            fig_s.add_trace(go.Scatter(
                x=xs, y=ys, 
                mode="lines", 
                name="Model Fit",
                line=dict(color="#0F172A", width=2, dash="dash")
            ))
        
        # Add spot price line
        fig_s.add_vline(x=spot, line_dash="dot", line_color="#F59E0B", 
                        annotation_text=f"Spot ${spot:.0f}", annotation_position="top")
        
        # Shade mispricing zones
        if "edge" in chain.columns and surface.valid:
            # Find best opportunities
            best_calls = calls_data.nlargest(3, "edge") if not calls_data.empty else pd.DataFrame()
            best_puts = puts_data.nlargest(3, "edge") if not puts_data.empty else pd.DataFrame()
            
            # Highlight zones with positive edge
            for _, row in best_calls.iterrows():
                if row["edge"] > 0.1:
                    fig_s.add_vrect(
                        x0=row["strike"] - 2, x1=row["strike"] + 2,
                        fillcolor="rgba(16, 185, 129, 0.15)",
                        line_width=0,
                        annotation_text=f"+${row['edge']:.2f}",
                        annotation_position="top left"
                    )
            
            for _, row in best_puts.iterrows():
                if row["edge"] > 0.1:
                    fig_s.add_vrect(
                        x0=row["strike"] - 2, x1=row["strike"] + 2,
                        fillcolor="rgba(16, 185, 129, 0.15)",
                        line_width=0
                    )
        
        fig_s.update_layout(
            template="plotly_white", 
            height=500,
            title=dict(
                text="<b>Volatility Skew</b> — Hover for details | Size = Edge magnitude | Green = Underpriced | Red = Overpriced",
                font=dict(size=14)
            ),
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest"
        )
        
        st.plotly_chart(fig_s, use_container_width=True)
        
        # Summary stats
        if "edge" in chain.columns:
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            pos_edge = chain[chain["edge"] > 0]
            neg_edge = chain[chain["edge"] < 0]
            
            with col_stats1:
                st.metric("Underpriced Options", len(pos_edge), help="Options trading below model fair value")
            with col_stats2:
                st.metric("Overpriced Options", len(neg_edge), help="Options trading above model fair value")
            with col_stats3:
                avg_call_iv = calls_data["impliedVolatility"].mean() if not calls_data.empty else 0
                avg_put_iv = puts_data["impliedVolatility"].mean() if not puts_data.empty else 0
                skew_val = avg_put_iv - avg_call_iv
                st.metric("Put-Call Skew", f"{skew_val:.1%}", 
                         delta="Puts richer" if skew_val > 0 else "Calls richer",
                         delta_color="inverse" if skew_val > 0.02 else "normal")
            with col_stats4:
                best_edge = chain["edge"].max()
                best_strike = chain.loc[chain["edge"].idxmax(), "strike"] if best_edge > 0 else "N/A"
                st.metric("Best Opportunity", f"${best_edge:.2f}", delta=f"@ ${best_strike}" if best_edge > 0 else None)

        # ==========================================
        # OPTIONS STRUCTURING ENGINE (Execution-Reality Aware)
        # ==========================================
        st.markdown("---")
        st.subheader("🏗️ Options Structuring Engine")
        st.caption("Execution-adjusted pricing • Liquidity-filtered • Slippage-aware")
        
        with st.expander("⚙️ Strategy Builder", expanded=False):
            str_col1, str_col2, str_col3 = st.columns(3)
            
            with str_col1:
                strategy_type = st.selectbox("Strategy Type", [
                    "Long Call", "Long Put",
                    "Bull Call Spread", "Bear Put Spread",
                    "Bull Put Spread (Credit)", "Bear Call Spread (Credit)",
                    "Long Straddle", "Long Strangle",
                    "Iron Condor", "Iron Butterfly",
                    "Calendar Spread", "Diagonal Spread"
                ])
            
            with str_col2:
                direction = st.selectbox("Market View", ["Bullish", "Bearish", "Neutral", "High Volatility", "Low Volatility"])
            
            with str_col3:
                max_risk = st.number_input("Max Risk ($)", value=500, min_value=100, step=100)
            
            # Liquidity & Execution Settings
            liq_col1, liq_col2 = st.columns(2)
            with liq_col1:
                min_pop = st.slider("Minimum Probability of Profit", 0.3, 0.8, 0.5)
            with liq_col2:
                max_spread_pct = st.slider("Max Bid-Ask Spread %", 0.05, 0.30, 0.15, help="Reject options with wider spreads")
            
            # TRADER MODE SETTINGS
            st.markdown("**🎯 Trade Selection Mode:**")
            mode_col1, mode_col2 = st.columns(2)
            with mode_col1:
                trader_mode = st.radio(
                    "Selection Mode",
                    ["Trader (Convexity)", "Capital Substitution"],
                    index=0,
                    help="Trader mode prioritizes convexity & non-linear payoffs. Capital sub allows deep ITM equity-like structures."
                )
            with mode_col2:
                require_convexity = st.checkbox("Require Convexity", value=True, 
                    help="Reject trades with near-linear payoffs")
                strike_proximity = st.slider("Max Strike Distance %", 0.15, 0.50, 0.30,
                    help="At least one strike must be within this % of spot")
            
            if st.button("🔍 Find Executable Structure", type="primary"):
                with st.spinner("Filtering for liquid, executable trades..."):
                    
                    # Get ATM strike
                    atm_strike = min(chain["strike"].unique(), key=lambda x: abs(x - spot))
                    
                    # ============ EXECUTION REALITY FILTERS ============
                    # Calculate bid-ask spread percentage for each option
                    exec_chain = chain.copy()
                    exec_chain["spread"] = exec_chain["ask"] - exec_chain["bid"]
                    exec_chain["spread_pct"] = exec_chain["spread"] / exec_chain["mid"].replace(0, np.nan)
                    exec_chain["spread_pct"] = exec_chain["spread_pct"].fillna(1.0)
                    
                    # HARD LIQUIDITY GATE: Reject illiquid options
                    liquid_chain = exec_chain[
                        (exec_chain["spread_pct"] <= max_spread_pct) &  # Spread filter
                        ((exec_chain["volume"] >= 5) | (exec_chain["openInterest"] >= 50)) &  # Activity filter
                        (exec_chain["bid"] > 0.01)  # Must have real bid
                    ].copy()
                    
                    if liquid_chain.empty:
                        st.error("❌ No liquid options found. All contracts fail execution filters (wide spreads or no volume).")
                        st.info("Try: Different expiration, closer to ATM strikes, or relax spread threshold.")
                        st.stop()
                    
                    # Execution-adjusted pricing: LONG = ASK, SHORT = BID
                    liquid_chain["exec_buy"] = liquid_chain["ask"]  # Price to buy (long)
                    liquid_chain["exec_sell"] = liquid_chain["bid"]  # Price to sell (short)
                    liquid_chain["slippage"] = liquid_chain["spread"] / 2  # Half-spread slippage estimate
                    
                    calls = liquid_chain[liquid_chain["type"] == "call"].sort_values("strike")
                    puts = liquid_chain[liquid_chain["type"] == "put"].sort_values("strike")
                    
                    # Track rejection reasons
                    rejection_reasons = []
                    
                    recommendation = None
                    legs = []
                    rationale = ""
                    risk_profile = ""
                    exec_warning = ""
                    trade_viable = False
                    
                    # ============================================================================
                    # CAPITAL ALLOCATOR ENGINE: EV FORMULAS, PROBABILITY PROXIES, QUALITY SCORING
                    # ============================================================================
                    
                    # Calculate time to expiration
                    dt_struct = (datetime.strptime(sel_exp, "%Y-%m-%d").date() - date.today()).days
                    T = max(dt_struct / 365.0, 0.001)  # Time in years
                    
                    # Get ATM implied volatility from chain for probability calculations
                    atm_options = liquid_chain[liquid_chain["strike"] == atm_strike]
                    if not atm_options.empty:
                        mkt_iv_calc = atm_options["impliedVolatility"].mean()
                    else:
                        # Fallback: use median IV from near-ATM options
                        near_atm = liquid_chain[(liquid_chain["strike"] >= spot * 0.95) & (liquid_chain["strike"] <= spot * 1.05)]
                        mkt_iv_calc = near_atm["impliedVolatility"].median() if not near_atm.empty else 0.30
                    
                    mkt_iv_calc = max(mkt_iv_calc, 0.10)  # Floor at 10% IV
                    
                    # Implied move for probability calculations
                    implied_move = spot * mkt_iv_calc * np.sqrt(T)
                    
                    # ----------------------------------------------
                    # 1. PROBABILITY PROXIES (When distribution sparse)
                    # ----------------------------------------------
                    
                    def prob_implied_move_proxy(strike, spot, implied_move):
                        """P(K) ≈ 1 - Φ(|K - S| / Implied_Move) - Conservative estimate."""
                        if implied_move <= 0:
                            return 0.5
                        z = abs(strike - spot) / implied_move
                        return 1 - norm.cdf(z)
                    
                    def prob_delta_proxy(delta, is_long_dated=False, has_skew=False):
                        """Use |delta| as probability, with penalties for skew/tenor."""
                        prob = abs(delta) if delta else 0.5
                        # Penalty for long-dated options (delta less reliable)
                        if is_long_dated:
                            prob *= 0.9
                        # Penalty for skew (OTM deltas less reliable)
                        if has_skew:
                            prob *= 0.95
                        return min(prob, 0.95)  # Cap at 95%
                    
                    def get_probability_confidence(delta, spread_pct, T):
                        """Score confidence in probability estimate (0-1)."""
                        confidence = 1.0
                        # Lower confidence for far OTM (low delta)
                        if abs(delta) < 0.2:
                            confidence *= 0.7
                        # Lower confidence for wide spreads (illiquid = bad price discovery)
                        if spread_pct > 0.15:
                            confidence *= 0.8
                        # Lower confidence for long-dated
                        if T > 0.25:  # > 3 months
                            confidence *= 0.85
                        return confidence
                    
                    # ----------------------------------------------
                    # VERTICAL SPREAD PROBABILITY HELPERS
                    # ----------------------------------------------
                    
                    def clamp_probability(p, T_years=0.1):
                        """
                        Clamp probabilities to avoid 0% or 100%.
                        Tighter bounds for long-dated trades to reflect tail risk.
                        """
                        if T_years > 0.25:  # > 3 months
                            return min(max(p, 0.02), 0.98)
                        elif T_years > 0.5:  # > 6 months
                            return min(max(p, 0.05), 0.95)
                        else:
                            return min(max(p, 0.01), 0.99)
                    
                    def apply_tail_risk_floor(prob_max_profit, prob_loss, T_years, is_deep_itm=False):
                        """
                        Apply tail-risk sanity floor.
                        Deep ITM spreads still have non-zero tail risk.
                        """
                        # Base tail risk epsilon (increases with time)
                        epsilon = 0.005 + (T_years * 0.02)  # 0.5% to 2%+ based on tenor
                        
                        # Higher tail risk for deep ITM structures
                        if is_deep_itm:
                            epsilon = max(epsilon, 0.02)  # At least 2% tail risk
                        
                        # Adjust probabilities
                        adj_prob_max_profit = prob_max_profit * (1 - epsilon)
                        adj_prob_loss = prob_loss + (epsilon * 0.5)  # Add tail component
                        
                        return adj_prob_max_profit, adj_prob_loss, epsilon
                    
                    def is_deep_itm_spread(spot, long_strike, short_strike, spread_type="call"):
                        """
                        Detect if spread is deep ITM (short strike far from spot).
                        """
                        if spread_type == "call":
                            # Bull call: deep ITM if short strike << spot
                            distance_pct = (spot - short_strike) / spot
                            return distance_pct > 0.15  # Short strike >15% below spot
                        else:
                            # Bear put: deep ITM if short strike >> spot
                            distance_pct = (short_strike - spot) / spot
                            return distance_pct > 0.15
                    
                    def get_spread_risk_label(prob_max_profit, prob_loss, is_deep_itm, tail_epsilon):
                        """
                        Generate appropriate risk label for vertical spreads.
                        Never use "riskless" or "guaranteed".
                        """
                        if is_deep_itm and prob_max_profit > 0.85:
                            return "⚠️ HIGH-PROB/HIGH-TAIL-RISK", "Deep ITM structure with concentrated tail risk"
                        elif prob_max_profit > 0.90:
                            return "📈 HIGH PROBABILITY", f"Model-estimated (tail risk: {tail_epsilon:.1%})"
                        elif prob_max_profit > 0.70:
                            return "✅ FAVORABLE ODDS", "Probability supports thesis"
                        elif prob_max_profit > 0.50:
                            return "⚡ MODERATE ODDS", "Balanced risk/reward"
                        else:
                            return "⚠️ SPECULATIVE", "Low probability of max profit"
                    
                    # ==============================================================
                    # TRADER-FOCUSED TRADE SELECTION LOGIC
                    # ==============================================================
                    
                    def check_strike_proximity(strike, spot, max_distance_pct=0.30):
                        """
                        STRIKE PROXIMITY CONSTRAINT: Strike must be within ±30% of spot.
                        """
                        distance = abs(strike - spot) / spot
                        return distance <= max_distance_pct
                    
                    def check_spread_proximity(long_strike, short_strike, spot, max_distance_pct=0.30):
                        """
                        At least one primary strike must be within ±30% of spot.
                        """
                        long_ok = check_strike_proximity(long_strike, spot, max_distance_pct)
                        short_ok = check_strike_proximity(short_strike, spot, max_distance_pct)
                        return long_ok or short_ok
                    
                    def calc_convexity_score(delta, gamma, vega, spot, option_price):
                        """
                        CONVEXITY REQUIREMENT: Measure non-linear payoff characteristics.
                        Returns score 0-1 where higher = more convex.
                        """
                        # Normalize gamma by spot squared and price
                        gamma_contrib = abs(gamma) * (spot ** 2) / max(option_price * 100, 1) if gamma else 0
                        
                        # Normalize vega by price
                        vega_contrib = abs(vega) / max(option_price * 100, 1) if vega else 0
                        
                        # Delta linearity penalty: |delta| near 0 or 1 = less convex
                        delta_abs = abs(delta) if delta else 0.5
                        linearity_penalty = 1 - abs(2 * delta_abs - 1)  # Max at delta=0.5
                        
                        # Combined convexity score
                        convexity = (gamma_contrib * 0.4 + vega_contrib * 0.3 + linearity_penalty * 0.3)
                        return min(convexity, 1.0)
                    
                    def calc_spread_convexity(long_delta, short_delta, long_gamma, short_gamma, long_vega, short_vega):
                        """
                        Calculate net convexity for a spread.
                        Deep ITM spreads have near-zero convexity.
                        """
                        net_gamma = (long_gamma or 0) - (short_gamma or 0)
                        net_vega = (long_vega or 0) - (short_vega or 0)
                        net_delta = abs((long_delta or 0) - (short_delta or 0))
                        
                        # Convexity score based on net Greeks
                        gamma_score = min(abs(net_gamma) * 100, 1.0)  # Scale gamma
                        vega_score = min(abs(net_vega) / 10, 1.0)     # Scale vega
                        
                        # Penalty for synthetic-stock behavior (delta spread near 1)
                        synthetic_penalty = 1 - min(net_delta, 1.0)
                        
                        convexity = (gamma_score * 0.4 + vega_score * 0.4 + synthetic_penalty * 0.2)
                        return convexity
                    
                    def is_equity_substitute(long_delta, short_delta, net_gamma, net_vega):
                        """
                        Detect if spread behaves like synthetic stock (linear payoff).
                        """
                        net_delta = abs((long_delta or 0) - (short_delta or 0))
                        # Synthetic if: high delta, near-zero gamma/vega
                        if net_delta > 0.85 and abs(net_gamma or 0) < 0.001 and abs(net_vega or 0) < 0.5:
                            return True
                        return False
                    
                    def classify_thesis(strategy_type, direction, spot, long_strike, short_strike=None, iv_rank=None):
                        """
                        THESIS CLASSIFICATION: Categorize the trade thesis.
                        Returns (category, one_sentence_thesis)
                        """
                        if strategy_type in ["Long Call", "Bull Call Spread", "Bull Put Spread (Credit)"]:
                            if short_strike and abs(long_strike - spot) < spot * 0.05:
                                thesis = "Bullish with upside convexity from near-ATM gamma exposure"
                                category = "Directional + Convexity"
                            else:
                                thesis = "Bullish directional with defined risk"
                                category = "Directional"
                        elif strategy_type in ["Long Put", "Bear Put Spread", "Bear Call Spread (Credit)"]:
                            if short_strike and abs(long_strike - spot) < spot * 0.05:
                                thesis = "Bearish with downside convexity from near-ATM gamma"
                                category = "Directional + Convexity"
                            else:
                                thesis = "Bearish directional with defined risk"
                                category = "Directional"
                        elif strategy_type in ["Long Straddle", "Long Strangle"]:
                            thesis = "Long volatility via dual-sided gamma/vega exposure"
                            category = "Volatility"
                        elif strategy_type in ["Iron Condor"]:
                            thesis = "Range-bound thesis with time decay capture"
                            category = "Volatility (Short)"
                        elif strategy_type in ["Iron Butterfly"]:
                            thesis = "Pin risk thesis with max profit at ATM strike"
                            category = "Volatility (Short)"
                        elif strategy_type in ["Calendar Spread"]:
                            thesis = "Term-structure play exploiting front/back IV differential"
                            category = "Volatility"
                        elif strategy_type in ["Diagonal Spread"]:
                            if direction in ["Bullish"]:
                                thesis = "Bullish bias with volatility leverage via time-structure convexity"
                            elif direction in ["Bearish"]:
                                thesis = "Bearish bias with volatility leverage via time-structure convexity"
                            else:
                                thesis = "Directional + volatility play with term structure mismatch"
                            category = "Hybrid"
                        else:
                            thesis = "Defined risk structure"
                            category = "Hybrid"
                        
                        return category, thesis
                    
                    def calc_payoff_acceleration(prob_max_profit, prob_profit, max_profit, max_loss, breakeven, spot):
                        """
                        PAYOFF ACCELERATION TEST: Does P/L accelerate favorably?
                        Favors trades where incremental favorable movement causes accelerating returns.
                        """
                        # Distance to breakeven as % of spot
                        be_distance = abs(breakeven - spot) / spot
                        
                        # Acceleration score: how quickly do we reach max profit after breakeven?
                        profit_zone = (prob_profit - 0) if prob_profit > 0 else 0
                        max_zone = prob_max_profit
                        
                        # Ratio of max profit zone to any profit zone (higher = more accelerated)
                        if profit_zone > 0:
                            acceleration = max_zone / profit_zone
                        else:
                            acceleration = 0
                        
                        # R:R component
                        rr = max_profit / max_loss if max_loss > 0 else 0
                        
                        # Combined score favoring convex payoffs
                        score = acceleration * 0.5 + min(rr / 3, 0.5)
                        return min(score, 1.0)
                    
                    def get_trader_label(category, thesis, convexity_score, is_eq_sub):
                        """
                        Generate trader-centric labeling.
                        """
                        if is_eq_sub:
                            return "⚠️ EQUITY SUBSTITUTE", "Linear payoff - not trader-grade convexity"
                        elif convexity_score < 0.15:
                            return "⚠️ LOW CONVEXITY", "Near-linear payoff structure"
                        elif convexity_score > 0.5:
                            return f"🎯 {category.upper()}", thesis
                        else:
                            return f"✅ {category.upper()}", thesis
                    
                    # ==============================================================
                    # TRADER-MODE GENERATION GATES (HARD EXCLUSION RULES)
                    # ==============================================================
                    
                    def passes_trader_generation_gate(long_strike, short_strike, spot, max_distance_pct):
                        """
                        TRADER MODE GENERATION GATE: Hard exclusion before scoring.
                        At least ONE primary payoff-driving strike must be within max_distance_pct of spot.
                        If ALL strikes are deep ITM (below spot by more than threshold), reject.
                        """
                        long_distance = abs(long_strike - spot) / spot
                        short_distance = abs(short_strike - spot) / spot if short_strike else float('inf')
                        
                        # At least one strike must be within threshold
                        if long_distance <= max_distance_pct or short_distance <= max_distance_pct:
                            return True
                        return False
                    
                    def passes_near_atm_anchor_rule(long_strike, short_strike, spot, direction, is_calendar=False):
                        """
                        NEAR-ATM ANCHOR RULE: At least one long leg must be near-ATM or slightly OTM.
                        Deep ITM longs only allowed in calendar/diagonal structures.
                        """
                        atm_threshold = spot * 0.08  # 8% from ATM = near-ATM
                        
                        # Check if long leg is near-ATM
                        long_near_atm = abs(long_strike - spot) <= atm_threshold
                        
                        # Check if long leg is slightly OTM (directionally aligned)
                        if direction in ["Bullish", "High Volatility"]:
                            long_slightly_otm = long_strike > spot and (long_strike - spot) / spot <= 0.15
                        else:
                            long_slightly_otm = long_strike < spot and (spot - long_strike) / spot <= 0.15
                        
                        if long_near_atm or long_slightly_otm:
                            return True
                        
                        # Deep ITM long allowed only in calendar/diagonal with near-ATM short
                        if is_calendar and short_strike:
                            short_near_atm = abs(short_strike - spot) <= atm_threshold
                            if short_near_atm:
                                return True
                        
                        return False
                    
                    def check_calendar_conditions(front_iv, back_iv, front_expiry_days, back_expiry_days):
                        """
                        CALENDAR/DIAGONAL GENERATION CONDITIONS:
                        Returns True if conditions favor calendar/diagonal generation.
                        """
                        # Condition 1: Near-term IV lower than back-term IV
                        if front_iv < back_iv * 0.95:
                            return True, "Term structure favors calendar (back IV > front IV)"
                        
                        # Condition 2: Forward volatility > spot implied volatility
                        if back_expiry_days > front_expiry_days:
                            implied_fwd_var = (back_iv**2 * back_expiry_days - front_iv**2 * front_expiry_days) / (back_expiry_days - front_expiry_days)
                            if implied_fwd_var > 0:
                                implied_fwd_vol = np.sqrt(implied_fwd_var)
                                if implied_fwd_vol > front_iv * 1.05:
                                    return True, "Forward vol elevated vs spot IV"
                        
                        # Condition 3: Term structure monotonicity violated (unusual shape)
                        vol_ratio = back_iv / front_iv if front_iv > 0 else 1
                        if vol_ratio > 1.15 or vol_ratio < 0.85:
                            return True, "Term structure anomaly detected"
                        
                        return False, None
                    
                    def is_deep_itm_candidate(long_strike, short_strike, spot, spread_type):
                        """
                        Check if candidate is a deep ITM structure that should be excluded in trader mode.
                        """
                        if spread_type == "call":
                            # Bull call: deep ITM if both strikes significantly below spot
                            if long_strike < spot * 0.85 and (short_strike is None or short_strike < spot * 0.90):
                                return True
                        else:  # put
                            # Bear put: deep ITM if both strikes significantly above spot
                            if long_strike > spot * 1.15 and (short_strike is None or short_strike > spot * 1.10):
                                return True
                        return False
                    
                    def get_calendar_thesis(front_iv, back_iv, direction):
                        """Generate thesis for calendar/diagonal spreads."""
                        if back_iv > front_iv:
                            return "Volatility", "Long back-month vega with front-month theta capture via term structure mismatch"
                        elif direction in ["Bullish", "Bearish"]:
                            return "Hybrid", f"{direction} bias with volatility leverage from time-structure convexity"
                        else:
                            return "Volatility", "Neutral volatility play exploiting term structure"
                    
                    # ----------------------------------------------
                    # 2. ONE-LINE EV FORMULAS BY STRATEGY TYPE
                    # ----------------------------------------------
                    
                    def ev_long_option(prob_itm, avg_payoff, entry_cost, iv_premium_penalty=0):
                        """
                        EV ≈ (P_ITM × Avg_Payoff) − Entry_Cost
                        Penalty if implied vol > forecast vol
                        """
                        ev = (prob_itm * avg_payoff * 0.5) - entry_cost  # 0.5 = conservative avg payoff factor
                        ev -= iv_premium_penalty  # Reduce EV if IV seems elevated
                        return ev
                    
                    def ev_vertical_debit(p_above_short, p_below_long, spread_width, entry_debit):
                        """
                        EV ≈ (P_Above_Short − P_Below_Long) × Spread_Width × 100 − Entry_Debit × 100
                        Simplified proxy for debit spreads.
                        """
                        ev = (p_above_short - p_below_long) * spread_width * 100 - entry_debit * 100
                        return ev
                    
                    def ev_vertical_credit(credit, prob_profit, max_loss, tail_risk_elevated=False):
                        """
                        EV ≈ Credit × P_Profit − Max_Loss × (1 − P_Profit)
                        Penalty if short strike within expected move.
                        """
                        effective_max_loss = max_loss * 1.2 if tail_risk_elevated else max_loss
                        ev = (credit * prob_profit) - (effective_max_loss * (1 - prob_profit))
                        return ev
                    
                    def ev_iron_condor(credit, prob_inside, max_loss):
                        """
                        EV ≈ Credit × P_Inside_Range − Avg_Tail_Loss × (1 − P_Inside_Range)
                        Avg_Tail_Loss ≥ 0.7 × max_loss (conservative)
                        """
                        avg_tail_loss = max_loss * 0.7  # Conservative tail loss estimate
                        ev = (credit * prob_inside) - (avg_tail_loss * (1 - prob_inside))
                        return ev
                    
                    # ----------------------------------------------
                    # 3. TRADE QUALITY SCORE ("Cool but Dumb" Filter)
                    # ----------------------------------------------
                    
                    def calc_trade_quality_score(ev, max_loss, prob_profit, prob_confidence, liquidity_score, complexity=1):
                        """
                        Trade_Quality = (Normalized_EV) × (Probability_Confidence) × (Liquidity_Score) ÷ (Complexity_Penalty)
                        
                        Penalizes:
                        - Low PoP
                        - High capital at risk
                        - Large slippage relative to payoff
                        - Complexity without proportional EV
                        """
                        if max_loss <= 0 or ev is None:
                            return 0
                        
                        # Normalize EV: EV per dollar risked, capped
                        normalized_ev = min(ev / max_loss, 0.5) if max_loss > 0 else 0
                        if normalized_ev < 0:
                            normalized_ev = normalized_ev * 2  # Penalize negative EV more
                        
                        # PoP penalty: trades with < 40% PoP get heavily penalized
                        pop_factor = prob_profit if prob_profit >= 0.4 else prob_profit * 0.5
                        
                        # Complexity penalty (1 = simple, 2 = spread, 4 = iron condor/butterfly)
                        complexity_penalty = 1 + (complexity - 1) * 0.15
                        
                        quality = (normalized_ev + 0.1) * prob_confidence * liquidity_score * pop_factor / complexity_penalty
                        
                        return max(0, quality)
                    
                    def get_trade_label(quality_score, ev, prob_profit, prob_confidence):
                        """Generate appropriate trade label based on quality metrics."""
                        if quality_score < 0.02:
                            return "❌ NOT RECOMMENDED", "Structurally Valid, Strategically Weak"
                        elif ev <= 0:
                            return "⚠️ NEGATIVE EV", "Expected value ≤ $0 after execution"
                        elif prob_profit < 0.35:
                            return "⚠️ LOW WIN RATE", "High Variance / Low Win Rate"
                        elif prob_confidence < 0.6:
                            return "⚠️ LOW CONFIDENCE", "Probability estimates unreliable"
                        elif quality_score < 0.05:
                            return "⚡ MARGINAL", "Positive EV but thin edge"
                        elif quality_score < 0.10:
                            return "✅ ACCEPTABLE", "Modest but robust"
                        else:
                            return "🎯 FAVORABLE", "Positive EV with good probability support"
                    
                    def should_recommend_no_trade(ev, prob_confidence, slippage_ratio, quality_score):
                        """
                        EXPLICIT OVERRIDE: If conditions are poor, recommend NO TRADE.
                        This is the CORRECT output when:
                        - EV is marginal (< $0.10 per contract)
                        - Probability confidence is low (< 0.5)
                        - Execution costs dominate (slippage > 20% of potential profit)
                        """
                        if ev is not None and ev < 0.10:
                            return True, "EV too marginal to justify execution risk"
                        if prob_confidence < 0.5:
                            return True, "Probability estimates too uncertain"
                        if slippage_ratio > 0.20:
                            return True, "Execution costs dominate potential profit"
                        if quality_score < 0.02:
                            return True, "Trade quality below minimum threshold"
                        return False, ""
                    
                    # ----------------------------------------------
                    # TRADE METRICS STORAGE
                    # ----------------------------------------------
                    trade_metrics = {
                        "ev_dollars": None,           # PRIMARY: Slippage-adjusted EV in $
                        "ev_per_dollar_risked": None, # DEFINED EDGE: EV / max risk
                        "prob_profit": None,          # Probability of any profit
                        "prob_max_profit": None,      # Probability of reaching max profit
                        "prob_breakeven": None,       # Probability of reaching breakeven
                        "prob_confidence": None,      # Confidence in probability estimate (0-1)
                        "liquidity_score": None,      # Execution quality score
                        "quality_score": None,        # Trade Quality Score for ranking
                        "trade_label": None,          # Human-readable quality label
                        "trade_sublabel": None,       # Detailed quality description
                        "complexity": 1,              # Structure complexity (1=simple, 4=iron condor)
                        "max_profit": None,           # INFORMATIONAL ONLY
                        "max_loss": None,             # INFORMATIONAL ONLY
                        "rr_ratio": None,             # INFORMATIONAL ONLY
                        "no_trade_reason": None,      # If NO TRADE recommended, why
                    }
                    
                    # ============ SINGLE LEG STRATEGIES (Capital Allocator Logic) ============
                    if strategy_type == "Long Call":
                        candidates = calls[(calls["strike"] >= spot * 0.95) & (calls["strike"] <= spot * 1.15)]
                        if not candidates.empty:
                            candidates = candidates.copy()
                            
                            # Calculate EV and Quality for each candidate
                            ev_results = []
                            for idx, row in candidates.iterrows():
                                exec_cost = row["exec_buy"]
                                delta = row.get("delta", 0.5)
                                
                                # PROBABILITY PROXY: Delta with confidence scoring
                                is_long_dated = T > 0.25
                                prob_itm = prob_delta_proxy(delta, is_long_dated=is_long_dated)
                                prob_conf = get_probability_confidence(delta, row["spread_pct"], T)
                                
                                # IV PREMIUM PENALTY: If IV seems elevated vs historical
                                iv_penalty = max(0, (row["impliedVolatility"] - mkt_iv_calc) * exec_cost * 0.5)
                                
                                # EV FORMULA: Long Option
                                avg_payoff = spot * row["impliedVolatility"] * np.sqrt(T)  # Expected move
                                ev = ev_long_option(prob_itm, avg_payoff, exec_cost, iv_penalty)
                                ev_per_risk = ev / exec_cost if exec_cost > 0 else 0
                                
                                liquidity = 1 - row["spread_pct"]
                                slippage_ratio = row["slippage"] / avg_payoff if avg_payoff > 0 else 1
                                
                                # TRADE QUALITY SCORE
                                quality = calc_trade_quality_score(
                                    ev=ev, max_loss=exec_cost, prob_profit=prob_itm,
                                    prob_confidence=prob_conf, liquidity_score=liquidity, complexity=1
                                )
                                
                                ev_results.append({
                                    "idx": idx, "ev_dollars": ev, "ev_per_risk": ev_per_risk,
                                    "prob_profit": prob_itm, "prob_conf": prob_conf,
                                    "exec_cost": exec_cost, "liquidity": liquidity,
                                    "quality": quality, "slippage_ratio": slippage_ratio
                                })
                            
                            ev_df = pd.DataFrame(ev_results)
                            
                            # RANKING: Sort by QUALITY SCORE (not just EV)
                            ev_df = ev_df.sort_values("quality", ascending=False)
                            best_ev = ev_df.iloc[0]
                            best = candidates.loc[best_ev["idx"]]
                            
                            # NO TRADE CHECK
                            no_trade, no_trade_reason = should_recommend_no_trade(
                                best_ev["ev_dollars"], best_ev["prob_conf"], 
                                best_ev["slippage_ratio"], best_ev["quality"]
                            )
                            
                            if no_trade:
                                rejection_reasons.append(f"NO TRADE RECOMMENDED: {no_trade_reason}")
                            elif best_ev["ev_dollars"] > 0:
                                exec_cost = best["exec_buy"]
                                breakeven = best["strike"] + exec_cost
                                prob_be = 1 - norm.cdf((np.log(breakeven/spot) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T)))
                                
                                legs = [{
                                    "type": "CALL", "strike": best["strike"], "direction": "LONG",
                                    "bid": best["bid"], "ask": best["ask"], "exec_price": exec_cost,
                                    "iv": best["impliedVolatility"]
                                }]
                                
                                # Generate quality label
                                label, sublabel = get_trade_label(
                                    best_ev["quality"], best_ev["ev_dollars"], 
                                    best_ev["prob_profit"], best_ev["prob_conf"]
                                )
                                
                                trade_metrics["ev_dollars"] = best_ev["ev_dollars"]
                                trade_metrics["ev_per_dollar_risked"] = best_ev["ev_per_risk"]
                                trade_metrics["prob_profit"] = best_ev["prob_profit"]
                                trade_metrics["prob_breakeven"] = prob_be
                                trade_metrics["prob_confidence"] = best_ev["prob_conf"]
                                trade_metrics["liquidity_score"] = best_ev["liquidity"]
                                trade_metrics["quality_score"] = best_ev["quality"]
                                trade_metrics["trade_label"] = label
                                trade_metrics["trade_sublabel"] = sublabel
                                trade_metrics["complexity"] = 1
                                trade_metrics["max_loss"] = exec_cost * 100
                                
                                trade_viable = True
                            else:
                                rejection_reasons.append("All calls have EV ≤ $0 after execution costs")
                    
                    elif strategy_type == "Long Put":
                        candidates = puts[(puts["strike"] >= spot * 0.85) & (puts["strike"] <= spot * 1.05)]
                        if not candidates.empty:
                            candidates = candidates.copy()
                            
                            ev_results = []
                            for idx, row in candidates.iterrows():
                                exec_cost = row["exec_buy"]
                                delta = row.get("delta", -0.5)
                                
                                is_long_dated = T > 0.25
                                prob_itm = prob_delta_proxy(delta, is_long_dated=is_long_dated)
                                prob_conf = get_probability_confidence(delta, row["spread_pct"], T)
                                
                                iv_penalty = max(0, (row["impliedVolatility"] - mkt_iv_calc) * exec_cost * 0.5)
                                avg_payoff = spot * row["impliedVolatility"] * np.sqrt(T)
                                ev = ev_long_option(prob_itm, avg_payoff, exec_cost, iv_penalty)
                                ev_per_risk = ev / exec_cost if exec_cost > 0 else 0
                                
                                liquidity = 1 - row["spread_pct"]
                                slippage_ratio = row["slippage"] / avg_payoff if avg_payoff > 0 else 1
                                
                                quality = calc_trade_quality_score(
                                    ev=ev, max_loss=exec_cost, prob_profit=prob_itm,
                                    prob_confidence=prob_conf, liquidity_score=liquidity, complexity=1
                                )
                                
                                ev_results.append({
                                    "idx": idx, "ev_dollars": ev, "ev_per_risk": ev_per_risk,
                                    "prob_profit": prob_itm, "prob_conf": prob_conf,
                                    "exec_cost": exec_cost, "liquidity": liquidity,
                                    "quality": quality, "slippage_ratio": slippage_ratio
                                })
                            
                            ev_df = pd.DataFrame(ev_results)
                            ev_df = ev_df.sort_values("quality", ascending=False)
                            best_ev = ev_df.iloc[0]
                            best = candidates.loc[best_ev["idx"]]
                            
                            no_trade, no_trade_reason = should_recommend_no_trade(
                                best_ev["ev_dollars"], best_ev["prob_conf"], 
                                best_ev["slippage_ratio"], best_ev["quality"]
                            )
                            
                            if no_trade:
                                rejection_reasons.append(f"NO TRADE RECOMMENDED: {no_trade_reason}")
                            elif best_ev["ev_dollars"] > 0:
                                exec_cost = best["exec_buy"]
                                breakeven = best["strike"] - exec_cost
                                prob_be = norm.cdf((np.log(breakeven/spot) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T)))
                                
                                legs = [{
                                    "type": "PUT", "strike": best["strike"], "direction": "LONG",
                                    "bid": best["bid"], "ask": best["ask"], "exec_price": exec_cost,
                                    "iv": best["impliedVolatility"]
                                }]
                                
                                label, sublabel = get_trade_label(
                                    best_ev["quality"], best_ev["ev_dollars"], 
                                    best_ev["prob_profit"], best_ev["prob_conf"]
                                )
                                
                                trade_metrics["ev_dollars"] = best_ev["ev_dollars"]
                                trade_metrics["ev_per_dollar_risked"] = best_ev["ev_per_risk"]
                                trade_metrics["prob_profit"] = best_ev["prob_profit"]
                                trade_metrics["prob_breakeven"] = prob_be
                                trade_metrics["prob_confidence"] = best_ev["prob_conf"]
                                trade_metrics["liquidity_score"] = best_ev["liquidity"]
                                trade_metrics["quality_score"] = best_ev["quality"]
                                trade_metrics["trade_label"] = label
                                trade_metrics["trade_sublabel"] = sublabel
                                trade_metrics["complexity"] = 1
                                trade_metrics["max_loss"] = exec_cost * 100
                                
                                trade_viable = True
                            else:
                                rejection_reasons.append("All puts have EV ≤ $0 after execution costs")
                    
                    # ============ VERTICAL SPREADS (Trader-Focused Logic) ============
                    elif strategy_type == "Bull Call Spread":
                        spread_candidates = []
                        trader_mode_active = trader_mode == "Trader (Convexity)"
                        candidates_rejected_trader_gate = 0
                        
                        for _, long_call in calls[calls["strike"] <= spot * 1.05].iterrows():
                            for _, short_call in calls[calls["strike"] > long_call["strike"]].iterrows():
                                if short_call["strike"] - long_call["strike"] > spot * 0.20:
                                    continue
                                
                                K_long = long_call["strike"]
                                K_short = short_call["strike"]
                                
                                # ========================================
                                # TRADER-MODE GENERATION GATES (HARD EXCLUSION)
                                # ========================================
                                
                                if trader_mode_active:
                                    # GATE 1: Deep ITM exclusion - DO NOT GENERATE
                                    if is_deep_itm_candidate(K_long, K_short, spot, "call"):
                                        candidates_rejected_trader_gate += 1
                                        continue  # HARD EXCLUSION - not even generated
                                    
                                    # GATE 2: Strike proximity - at least one strike near spot
                                    if not passes_trader_generation_gate(K_long, K_short, spot, strike_proximity):
                                        candidates_rejected_trader_gate += 1
                                        continue  # HARD EXCLUSION
                                    
                                    # GATE 3: Near-ATM anchor rule
                                    if not passes_near_atm_anchor_rule(K_long, K_short, spot, direction):
                                        candidates_rejected_trader_gate += 1
                                        continue  # HARD EXCLUSION
                                
                                exec_debit = long_call["exec_buy"] - short_call["exec_sell"]
                                width = short_call["strike"] - long_call["strike"]
                                max_profit = width - exec_debit
                                max_loss = exec_debit
                                
                                if max_loss <= 0 or max_profit <= 0:
                                    continue
                                total_slippage = long_call["slippage"] + short_call["slippage"]
                                if exec_debit < total_slippage:
                                    continue
                                
                                rr_ratio = max_profit / max_loss
                                if rr_ratio > 10:
                                    continue
                                
                                # 2. CONVEXITY REQUIREMENT
                                long_gamma = long_call.get("gamma", 0.01)
                                short_gamma = short_call.get("gamma", 0.005)
                                long_vega = long_call.get("vega", 0.1)
                                short_vega = short_call.get("vega", 0.05)
                                long_delta = long_call.get("delta", 0.5)
                                short_delta = short_call.get("delta", 0.3)
                                
                                convexity_score = calc_spread_convexity(
                                    long_delta, short_delta, long_gamma, short_gamma, long_vega, short_vega
                                )
                                
                                # Check for equity substitute (linear payoff)
                                net_gamma = (long_gamma or 0) - (short_gamma or 0)
                                net_vega = (long_vega or 0) - (short_vega or 0)
                                is_eq_sub = is_equity_substitute(long_delta, short_delta, net_gamma, net_vega)
                                
                                # In trader mode, reject equity substitutes and low convexity
                                if trader_mode_active and require_convexity:
                                    if is_eq_sub:
                                        continue  # Skip equity substitute
                                    if convexity_score < 0.10:
                                        continue  # Skip low convexity
                                
                                # 3. THESIS CLASSIFICATION
                                category, thesis = classify_thesis(
                                    strategy_type, direction, spot, K_long, K_short
                                )
                                
                                # ========================================
                                # VERTICAL PROBABILITY DEFINITIONS (EXACT)
                                # ========================================
                                breakeven = K_long + exec_debit
                                
                                # P(Max Profit) = P(S_T >= K_short) - probability spot finishes above short strike
                                d2_short = (np.log(spot/K_short) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                p_max_profit_raw = norm.cdf(d2_short)
                                
                                # P(Max Loss) = P(S_T <= K_long) - probability spot finishes below long strike
                                d2_long = (np.log(spot/K_long) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                p_max_loss_raw = 1 - norm.cdf(d2_long)
                                
                                # P(Profit) = P(S_T >= breakeven) - probability spot finishes above breakeven
                                d2_be = (np.log(spot/breakeven) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                p_profit_raw = norm.cdf(d2_be)
                                
                                # DEEP ITM CHECK
                                deep_itm = is_deep_itm_spread(spot, K_long, K_short, "call")
                                
                                # APPLY TAIL RISK FLOOR (no 0% or 100%)
                                p_max_profit, p_max_loss, tail_eps = apply_tail_risk_floor(
                                    p_max_profit_raw, p_max_loss_raw, T, is_deep_itm=deep_itm
                                )
                                
                                # CLAMP ALL PROBABILITIES (never 0% or 100%)
                                p_max_profit = clamp_probability(p_max_profit, T)
                                p_max_loss = clamp_probability(p_max_loss, T)
                                prob_profit = clamp_probability(p_profit_raw, T)
                                
                                # SANITY: P(Profit) must be >= P(Max Profit)
                                prob_profit = max(prob_profit, p_max_profit)
                                
                                # EV FORMULA: Vertical Debit Spread
                                ev = ev_vertical_debit(p_max_profit, p_max_loss, width, exec_debit)
                                ev_per_risk = ev / (max_loss * 100) if max_loss > 0 else 0
                                
                                # PROBABILITY CONFIDENCE
                                avg_spread_pct = (long_call["spread_pct"] + short_call["spread_pct"]) / 2
                                avg_delta = (abs(long_call.get("delta", 0.5)) + abs(short_call.get("delta", 0.3))) / 2
                                prob_conf = get_probability_confidence(avg_delta, avg_spread_pct, T)
                                
                                # PENALIZE DEEP ITM (high prob but high tail risk, low edge)
                                if deep_itm:
                                    prob_conf *= 0.8  # Lower confidence for deep ITM
                                
                                liquidity = 1 - max(long_call["spread_pct"], short_call["spread_pct"])
                                slippage_ratio = total_slippage / max_profit if max_profit > 0 else 1
                                
                                # TRADE QUALITY SCORE (complexity = 2 for spreads)
                                # Penalize deep ITM structures
                                complexity_adj = 2.5 if deep_itm else 2
                                quality = calc_trade_quality_score(
                                    ev=ev/100, max_loss=max_loss, prob_profit=prob_profit,
                                    prob_confidence=prob_conf, liquidity_score=liquidity, complexity=complexity_adj
                                )
                                
                                # 4. PAYOFF ACCELERATION
                                payoff_accel = calc_payoff_acceleration(
                                    p_max_profit, prob_profit, max_profit, max_loss, breakeven, spot
                                )
                                
                                # TRADER-ADJUSTED QUALITY SCORE
                                # In trader mode: boost convexity, penalize equity substitutes
                                if trader_mode_active:
                                    convexity_boost = convexity_score * 0.3
                                    eq_sub_penalty = 0.5 if is_eq_sub else 0
                                    accel_boost = payoff_accel * 0.2
                                    quality = quality * (1 + convexity_boost + accel_boost - eq_sub_penalty)
                                
                                # Get risk label - use trader label if in trader mode
                                if trader_mode_active:
                                    risk_label, risk_sublabel = get_trader_label(category, thesis, convexity_score, is_eq_sub)
                                else:
                                    risk_label, risk_sublabel = get_spread_risk_label(p_max_profit, p_max_loss, deep_itm, tail_eps)
                                
                                spread_candidates.append({
                                    "long": long_call, "short": short_call,
                                    "exec_debit": exec_debit, "max_profit": max_profit, "max_loss": max_loss,
                                    "rr": rr_ratio, "slippage": total_slippage,
                                    "ev_dollars": ev, "ev_per_risk": ev_per_risk,
                                    "prob_profit": prob_profit, "prob_max_profit": p_max_profit,
                                    "prob_max_loss": p_max_loss, "tail_epsilon": tail_eps,
                                    "prob_conf": prob_conf, "liquidity": liquidity, "breakeven": breakeven,
                                    "quality": quality, "slippage_ratio": slippage_ratio,
                                    "is_deep_itm": deep_itm, "risk_label": risk_label, "risk_sublabel": risk_sublabel,
                                    "convexity_score": convexity_score, "is_eq_sub": is_eq_sub,
                                    "thesis_category": category, "thesis": thesis, "payoff_accel": payoff_accel
                                })
                        
                        if spread_candidates:
                            # RANKING: Sort by QUALITY SCORE (convexity-adjusted in trader mode)
                            spread_candidates.sort(key=lambda x: x["quality"], reverse=True)
                            best_spread = spread_candidates[0]
                            
                            # In trader mode, reject if best candidate is equity substitute
                            if trader_mode_active and best_spread.get("is_eq_sub"):
                                rejection_reasons.append("NO TRADE: Best candidate is equity substitute (linear payoff)")
                                spread_candidates = []
                            
                            # NO TRADE CHECK
                            no_trade, no_trade_reason = should_recommend_no_trade(
                                best_spread["ev_dollars"]/100, best_spread["prob_conf"],
                                best_spread["slippage_ratio"], best_spread["quality"]
                            )
                            
                            if no_trade:
                                rejection_reasons.append(f"NO TRADE RECOMMENDED: {no_trade_reason}")
                            elif best_spread["max_loss"] * 100 <= max_risk and best_spread["ev_dollars"] > 0:
                                legs = [
                                    {"type": "CALL", "strike": best_spread["long"]["strike"], "direction": "LONG", 
                                     "bid": best_spread["long"]["bid"], "ask": best_spread["long"]["ask"], "exec_price": best_spread["long"]["exec_buy"]},
                                    {"type": "CALL", "strike": best_spread["short"]["strike"], "direction": "SHORT",
                                     "bid": best_spread["short"]["bid"], "ask": best_spread["short"]["ask"], "exec_price": best_spread["short"]["exec_sell"]}
                                ]
                                
                                # Use trader-focused label or standard label
                                label = best_spread.get("risk_label", "✅ ACCEPTABLE")
                                sublabel = best_spread.get("risk_sublabel", "")
                                
                                trade_metrics["ev_dollars"] = best_spread["ev_dollars"] / 100
                                trade_metrics["ev_per_dollar_risked"] = best_spread["ev_per_risk"]
                                trade_metrics["prob_profit"] = best_spread["prob_profit"]
                                trade_metrics["prob_max_profit"] = best_spread["prob_max_profit"]
                                trade_metrics["prob_max_loss"] = best_spread.get("prob_max_loss", 0)
                                trade_metrics["prob_breakeven"] = best_spread["prob_profit"]
                                trade_metrics["prob_confidence"] = best_spread["prob_conf"]
                                trade_metrics["tail_epsilon"] = best_spread.get("tail_epsilon", 0.01)
                                trade_metrics["is_deep_itm"] = best_spread.get("is_deep_itm", False)
                                trade_metrics["is_eq_sub"] = best_spread.get("is_eq_sub", False)
                                trade_metrics["convexity_score"] = best_spread.get("convexity_score", 0.5)
                                trade_metrics["thesis_category"] = best_spread.get("thesis_category", "Directional")
                                trade_metrics["thesis"] = best_spread.get("thesis", "")
                                trade_metrics["payoff_accel"] = best_spread.get("payoff_accel", 0.5)
                                trade_metrics["liquidity_score"] = best_spread["liquidity"]
                                trade_metrics["quality_score"] = best_spread["quality"]
                                trade_metrics["trade_label"] = label
                                trade_metrics["trade_sublabel"] = sublabel
                                trade_metrics["complexity"] = 2
                                trade_metrics["max_profit"] = best_spread["max_profit"] * 100
                                trade_metrics["max_loss"] = best_spread["max_loss"] * 100
                                trade_metrics["rr_ratio"] = best_spread["rr"]
                                trade_metrics["breakeven"] = best_spread["breakeven"]
                                
                                trade_viable = True
                            elif best_spread["max_loss"] * 100 > max_risk:
                                rejection_reasons.append(f"Best spread exceeds max risk (${best_spread['max_loss']*100:.0f} > ${max_risk})")
                            else:
                                rejection_reasons.append("Best spread has EV ≤ $0")
                        else:
                            rejection_reasons.append("No bull call spreads passed quality filters")
                    
                    elif strategy_type == "Bear Put Spread":
                        spread_candidates = []
                        trader_mode_active = trader_mode == "Trader (Convexity)"
                        candidates_rejected_trader_gate = 0
                        
                        for _, long_put in puts[puts["strike"] >= spot * 0.95].iterrows():
                            for _, short_put in puts[puts["strike"] < long_put["strike"]].iterrows():
                                if long_put["strike"] - short_put["strike"] > spot * 0.20:
                                    continue
                                
                                K_long = long_put["strike"]
                                K_short = short_put["strike"]
                                
                                # ========================================
                                # TRADER-MODE GENERATION GATES (HARD EXCLUSION)
                                # ========================================
                                if trader_mode_active:
                                    # GATE 1: Deep ITM exclusion
                                    if is_deep_itm_candidate(K_long, K_short, spot, "put"):
                                        candidates_rejected_trader_gate += 1
                                        continue
                                    
                                    # GATE 2: Strike proximity
                                    if not passes_trader_generation_gate(K_long, K_short, spot, strike_proximity):
                                        candidates_rejected_trader_gate += 1
                                        continue
                                    
                                    # GATE 3: Near-ATM anchor rule
                                    if not passes_near_atm_anchor_rule(K_long, K_short, spot, direction):
                                        candidates_rejected_trader_gate += 1
                                        continue
                                
                                exec_debit = long_put["exec_buy"] - short_put["exec_sell"]
                                width = long_put["strike"] - short_put["strike"]
                                max_profit = width - exec_debit
                                max_loss = exec_debit
                                
                                if max_loss <= 0 or max_profit <= 0:
                                    continue
                                total_slippage = long_put["slippage"] + short_put["slippage"]
                                if exec_debit < total_slippage:
                                    continue
                                
                                rr_ratio = max_profit / max_loss
                                if rr_ratio > 10:
                                    continue
                                
                                # CONVEXITY CALCULATION
                                long_gamma = long_put.get("gamma", 0.01)
                                short_gamma = short_put.get("gamma", 0.005)
                                long_vega = long_put.get("vega", 0.1)
                                short_vega = short_put.get("vega", 0.05)
                                long_delta = long_put.get("delta", -0.5)
                                short_delta = short_put.get("delta", -0.3)
                                
                                convexity_score = calc_spread_convexity(
                                    long_delta, short_delta, long_gamma, short_gamma, long_vega, short_vega
                                )
                                net_gamma = (long_gamma or 0) - (short_gamma or 0)
                                net_vega = (long_vega or 0) - (short_vega or 0)
                                is_eq_sub = is_equity_substitute(long_delta, short_delta, net_gamma, net_vega)
                                
                                if trader_mode_active and require_convexity:
                                    if is_eq_sub or convexity_score < 0.10:
                                        continue
                                
                                category, thesis = classify_thesis(
                                    strategy_type, direction, spot, K_long, K_short
                                )
                                
                                # ========================================
                                # VERTICAL PROBABILITY DEFINITIONS (EXACT)
                                # ========================================
                                breakeven = K_long - exec_debit
                                
                                # P(Max Profit) = P(S_T <= K_short) - probability spot finishes below short strike
                                d2_short = (np.log(spot/K_short) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                p_max_profit_raw = 1 - norm.cdf(d2_short)
                                
                                # P(Max Loss) = P(S_T >= K_long) - probability spot finishes above long strike
                                d2_long = (np.log(spot/K_long) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                p_max_loss_raw = norm.cdf(d2_long)
                                
                                # P(Profit) = P(S_T <= breakeven)
                                d2_be = (np.log(spot/breakeven) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                p_profit_raw = 1 - norm.cdf(d2_be)
                                
                                # DEEP ITM CHECK (for puts, short strike >> spot means deep ITM)
                                deep_itm = is_deep_itm_spread(spot, K_long, K_short, "put")
                                
                                # APPLY TAIL RISK FLOOR
                                p_max_profit, p_max_loss, tail_eps = apply_tail_risk_floor(
                                    p_max_profit_raw, p_max_loss_raw, T, is_deep_itm=deep_itm
                                )
                                
                                # CLAMP ALL PROBABILITIES
                                p_max_profit = clamp_probability(p_max_profit, T)
                                p_max_loss = clamp_probability(p_max_loss, T)
                                prob_profit = clamp_probability(p_profit_raw, T)
                                
                                # SANITY: P(Profit) must be >= P(Max Profit)
                                prob_profit = max(prob_profit, p_max_profit)
                                
                                # EV CALCULATION
                                prob_between = max(0, 1 - p_max_profit - p_max_loss)
                                avg_partial_payoff = (max_profit / 2)
                                
                                ev = (p_max_profit * max_profit) + (prob_between * avg_partial_payoff) - (p_max_loss * max_loss)
                                ev_per_risk = ev / max_loss if max_loss > 0 else 0
                                
                                if ev <= 0:
                                    continue
                                
                                # PROBABILITY CONFIDENCE
                                avg_spread_pct = (long_put["spread_pct"] + short_put["spread_pct"]) / 2
                                avg_delta = (abs(long_put.get("delta", -0.5)) + abs(short_put.get("delta", -0.3))) / 2
                                prob_conf = get_probability_confidence(avg_delta, avg_spread_pct, T)
                                
                                if deep_itm:
                                    prob_conf *= 0.8
                                
                                liquidity = 1 - max(long_put["spread_pct"], short_put["spread_pct"])
                                slippage_ratio = total_slippage / max_profit if max_profit > 0 else 1
                                
                                complexity_adj = 2.5 if deep_itm else 2
                                
                                # Trader-adjusted quality
                                if trader_mode_active:
                                    convexity_boost = convexity_score * 0.3
                                    eq_sub_penalty = 0.5 if is_eq_sub else 0
                                    quality = calc_trade_quality_score(
                                        ev=ev/100, max_loss=max_loss, prob_profit=prob_profit,
                                        prob_confidence=prob_conf, liquidity_score=liquidity, complexity=complexity_adj
                                    ) * (1 + convexity_boost - eq_sub_penalty)
                                else:
                                    quality = calc_trade_quality_score(
                                        ev=ev/100, max_loss=max_loss, prob_profit=prob_profit,
                                        prob_confidence=prob_conf, liquidity_score=liquidity, complexity=complexity_adj
                                    )
                                
                                if trader_mode_active:
                                    risk_label, risk_sublabel = get_trader_label(category, thesis, convexity_score, is_eq_sub)
                                else:
                                    risk_label, risk_sublabel = get_spread_risk_label(p_max_profit, p_max_loss, deep_itm, tail_eps)
                                
                                spread_candidates.append({
                                    "long": long_put, "short": short_put,
                                    "exec_debit": exec_debit, "max_profit": max_profit, "max_loss": max_loss,
                                    "rr": rr_ratio, "slippage": total_slippage,
                                    "ev_dollars": ev, "ev_per_risk": ev_per_risk,
                                    "prob_profit": prob_profit, "prob_max_profit": p_max_profit,
                                    "prob_max_loss": p_max_loss, "tail_epsilon": tail_eps,
                                    "prob_conf": prob_conf, "liquidity": liquidity, "breakeven": breakeven,
                                    "quality": quality, "slippage_ratio": slippage_ratio,
                                    "is_deep_itm": deep_itm, "risk_label": risk_label, "risk_sublabel": risk_sublabel,
                                    "convexity_score": convexity_score, "is_eq_sub": is_eq_sub,
                                    "thesis_category": category, "thesis": thesis
                                })
                        
                        if spread_candidates:
                            spread_candidates.sort(key=lambda x: x["quality"], reverse=True)
                            best_spread = spread_candidates[0]
                            
                            # Reject if equity substitute in trader mode
                            if trader_mode_active and best_spread.get("is_eq_sub"):
                                rejection_reasons.append("NO TRADE: Best candidate is equity substitute (linear payoff)")
                                spread_candidates = []
                            elif best_spread["max_loss"] * 100 <= max_risk:
                                legs = [
                                    {"type": "PUT", "strike": best_spread["long"]["strike"], "direction": "LONG",
                                     "bid": best_spread["long"]["bid"], "ask": best_spread["long"]["ask"], "exec_price": best_spread["long"]["exec_buy"]},
                                    {"type": "PUT", "strike": best_spread["short"]["strike"], "direction": "SHORT",
                                     "bid": best_spread["short"]["bid"], "ask": best_spread["short"]["ask"], "exec_price": best_spread["short"]["exec_sell"]}
                                ]
                                
                                label = best_spread.get("risk_label", "✅ ACCEPTABLE")
                                sublabel = best_spread.get("risk_sublabel", "")
                                
                                trade_metrics["ev_dollars"] = best_spread["ev_dollars"]
                                trade_metrics["ev_per_dollar_risked"] = best_spread["ev_per_risk"]
                                trade_metrics["prob_profit"] = best_spread["prob_profit"]
                                trade_metrics["prob_max_profit"] = best_spread["prob_max_profit"]
                                trade_metrics["prob_max_loss"] = best_spread.get("prob_max_loss", 0)
                                trade_metrics["prob_breakeven"] = best_spread["prob_profit"]
                                trade_metrics["prob_confidence"] = best_spread["prob_conf"]
                                trade_metrics["tail_epsilon"] = best_spread.get("tail_epsilon", 0.01)
                                trade_metrics["is_deep_itm"] = best_spread.get("is_deep_itm", False)
                                trade_metrics["is_eq_sub"] = best_spread.get("is_eq_sub", False)
                                trade_metrics["convexity_score"] = best_spread.get("convexity_score", 0.5)
                                trade_metrics["thesis_category"] = best_spread.get("thesis_category", "Directional")
                                trade_metrics["thesis"] = best_spread.get("thesis", "")
                                trade_metrics["liquidity_score"] = best_spread["liquidity"]
                                trade_metrics["quality_score"] = best_spread["quality"]
                                trade_metrics["trade_label"] = label
                                trade_metrics["trade_sublabel"] = sublabel
                                trade_metrics["complexity"] = 2
                                trade_metrics["max_profit"] = best_spread["max_profit"] * 100
                                trade_metrics["max_loss"] = best_spread["max_loss"] * 100
                                trade_metrics["rr_ratio"] = best_spread["rr"]
                                trade_metrics["breakeven"] = best_spread["breakeven"]
                                
                                trade_viable = True
                            elif best_spread["max_loss"] * 100 > max_risk:
                                rejection_reasons.append(f"Best spread exceeds max risk (${best_spread['max_loss']*100:.0f} > ${max_risk})")
                            else:
                                rejection_reasons.append("Best spread has EV ≤ $0")
                        else:
                            rejection_reasons.append("No bear put spreads passed quality filters")
                    
                    # ============ CREDIT SPREADS (EV-Primary) ============
                    elif strategy_type == "Bull Put Spread (Credit)":
                        spread_candidates = []
                        
                        for _, short_put in puts[puts["strike"] <= spot * 0.95].iterrows():
                            for _, long_put in puts[puts["strike"] < short_put["strike"]].iterrows():
                                if short_put["strike"] - long_put["strike"] > spot * 0.15:
                                    continue
                                
                                exec_credit = short_put["exec_sell"] - long_put["exec_buy"]
                                width = short_put["strike"] - long_put["strike"]
                                max_loss = width - exec_credit
                                max_profit = exec_credit
                                
                                if max_loss <= 0 or max_profit <= 0:
                                    continue
                                total_slippage = short_put["slippage"] + long_put["slippage"]
                                if exec_credit < total_slippage:
                                    continue
                                
                                rr_ratio = max_profit / max_loss
                                if rr_ratio > 10:
                                    continue
                                
                                # PROBABILITY CALCULATIONS (model-implied)
                                # P(max profit) = P(spot > short strike at expiry)
                                d2_short = (np.log(spot/short_put["strike"]) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                prob_max_profit = norm.cdf(d2_short)
                                
                                # P(max loss) = P(spot < long strike)
                                d2_long = (np.log(spot/long_put["strike"]) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                prob_max_loss = 1 - norm.cdf(d2_long)
                                
                                # P(profit) = P(spot > breakeven = short - credit)
                                breakeven = short_put["strike"] - exec_credit
                                d2_be = (np.log(spot/breakeven) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                prob_profit = norm.cdf(d2_be)
                                
                                if prob_profit < min_pop:
                                    continue
                                
                                # EV CALCULATION
                                prob_between = max(0, 1 - prob_max_profit - prob_max_loss)
                                avg_partial_payoff = (max_profit - max_loss) / 2
                                
                                ev = (prob_max_profit * max_profit) + (prob_between * avg_partial_payoff) - (prob_max_loss * max_loss)
                                ev_per_risk = ev / max_loss if max_loss > 0 else 0
                                
                                if ev <= 0:
                                    continue
                                
                                liquidity = 1 - max(short_put["spread_pct"], long_put["spread_pct"])
                                
                                spread_candidates.append({
                                    "short": short_put, "long": long_put,
                                    "exec_credit": exec_credit, "max_profit": max_profit, "max_loss": max_loss,
                                    "rr": rr_ratio, "slippage": total_slippage,
                                    "ev_dollars": ev, "ev_per_risk": ev_per_risk,
                                    "prob_profit": prob_profit, "prob_max_profit": prob_max_profit,
                                    "prob_max_loss": prob_max_loss, "liquidity": liquidity, "breakeven": breakeven
                                })
                        
                        if spread_candidates:
                            spread_candidates.sort(key=lambda x: (x["ev_dollars"], x["prob_profit"], x["liquidity"]), reverse=True)
                            best_spread = spread_candidates[0]
                            
                            if best_spread["max_loss"] * 100 <= max_risk:
                                legs = [
                                    {"type": "PUT", "strike": best_spread["short"]["strike"], "direction": "SHORT",
                                     "bid": best_spread["short"]["bid"], "ask": best_spread["short"]["ask"], "exec_price": best_spread["short"]["exec_sell"]},
                                    {"type": "PUT", "strike": best_spread["long"]["strike"], "direction": "LONG",
                                     "bid": best_spread["long"]["bid"], "ask": best_spread["long"]["ask"], "exec_price": best_spread["long"]["exec_buy"]}
                                ]
                                
                                trade_metrics["ev_dollars"] = best_spread["ev_dollars"]
                                trade_metrics["ev_per_dollar_risked"] = best_spread["ev_per_risk"]
                                trade_metrics["prob_profit"] = best_spread["prob_profit"]
                                trade_metrics["prob_max_profit"] = best_spread["prob_max_profit"]
                                trade_metrics["prob_breakeven"] = best_spread["prob_profit"]
                                trade_metrics["liquidity_score"] = best_spread["liquidity"]
                                trade_metrics["max_profit"] = best_spread["max_profit"] * 100
                                trade_metrics["max_loss"] = best_spread["max_loss"] * 100
                                trade_metrics["rr_ratio"] = best_spread["rr"]
                                
                                trade_viable = True
                            else:
                                rejection_reasons.append(f"Best spread exceeds max risk")
                        else:
                            rejection_reasons.append("No bull put spreads with EV > $0 after execution costs")
                    
                    elif strategy_type == "Bear Call Spread (Credit)":
                        spread_candidates = []
                        
                        for _, short_call in calls[calls["strike"] >= spot * 1.05].iterrows():
                            for _, long_call in calls[calls["strike"] > short_call["strike"]].iterrows():
                                if long_call["strike"] - short_call["strike"] > spot * 0.15:
                                    continue
                                
                                exec_credit = short_call["exec_sell"] - long_call["exec_buy"]
                                width = long_call["strike"] - short_call["strike"]
                                max_loss = width - exec_credit
                                max_profit = exec_credit
                                
                                if max_loss <= 0 or max_profit <= 0:
                                    continue
                                total_slippage = short_call["slippage"] + long_call["slippage"]
                                if exec_credit < total_slippage:
                                    continue
                                
                                rr_ratio = max_profit / max_loss
                                if rr_ratio > 10:
                                    continue
                                
                                # PROBABILITY CALCULATIONS
                                # P(max profit) = P(spot < short strike at expiry)
                                d2_short = (np.log(spot/short_call["strike"]) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                prob_max_profit = 1 - norm.cdf(d2_short)
                                
                                # P(max loss) = P(spot > long strike)
                                d2_long = (np.log(spot/long_call["strike"]) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                prob_max_loss = norm.cdf(d2_long)
                                
                                # P(profit) = P(spot < breakeven = short + credit)
                                breakeven = short_call["strike"] + exec_credit
                                d2_be = (np.log(spot/breakeven) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                                prob_profit = 1 - norm.cdf(d2_be)
                                
                                if prob_profit < min_pop:
                                    continue
                                
                                # EV CALCULATION
                                prob_between = max(0, 1 - prob_max_profit - prob_max_loss)
                                avg_partial_payoff = (max_profit - max_loss) / 2
                                
                                ev = (prob_max_profit * max_profit) + (prob_between * avg_partial_payoff) - (prob_max_loss * max_loss)
                                ev_per_risk = ev / max_loss if max_loss > 0 else 0
                                
                                if ev <= 0:
                                    continue
                                
                                liquidity = 1 - max(short_call["spread_pct"], long_call["spread_pct"])
                                
                                spread_candidates.append({
                                    "short": short_call, "long": long_call,
                                    "exec_credit": exec_credit, "max_profit": max_profit, "max_loss": max_loss,
                                    "rr": rr_ratio, "slippage": total_slippage,
                                    "ev_dollars": ev, "ev_per_risk": ev_per_risk,
                                    "prob_profit": prob_profit, "prob_max_profit": prob_max_profit,
                                    "prob_max_loss": prob_max_loss, "liquidity": liquidity, "breakeven": breakeven
                                })
                        
                        if spread_candidates:
                            spread_candidates.sort(key=lambda x: (x["ev_dollars"], x["prob_profit"], x["liquidity"]), reverse=True)
                            best_spread = spread_candidates[0]
                            
                            if best_spread["max_loss"] * 100 <= max_risk:
                                legs = [
                                    {"type": "CALL", "strike": best_spread["short"]["strike"], "direction": "SHORT",
                                     "bid": best_spread["short"]["bid"], "ask": best_spread["short"]["ask"], "exec_price": best_spread["short"]["exec_sell"]},
                                    {"type": "CALL", "strike": best_spread["long"]["strike"], "direction": "LONG",
                                     "bid": best_spread["long"]["bid"], "ask": best_spread["long"]["ask"], "exec_price": best_spread["long"]["exec_buy"]}
                                ]
                                
                                trade_metrics["ev_dollars"] = best_spread["ev_dollars"]
                                trade_metrics["ev_per_dollar_risked"] = best_spread["ev_per_risk"]
                                trade_metrics["prob_profit"] = best_spread["prob_profit"]
                                trade_metrics["prob_max_profit"] = best_spread["prob_max_profit"]
                                trade_metrics["prob_breakeven"] = best_spread["prob_profit"]
                                trade_metrics["liquidity_score"] = best_spread["liquidity"]
                                trade_metrics["max_profit"] = best_spread["max_profit"] * 100
                                trade_metrics["max_loss"] = best_spread["max_loss"] * 100
                                trade_metrics["rr_ratio"] = best_spread["rr"]
                                
                                trade_viable = True
                            else:
                                rejection_reasons.append(f"Best spread exceeds max risk")
                        else:
                            rejection_reasons.append("No bear call spreads with EV > $0 after execution costs")
                    
                    # ============ VOLATILITY STRATEGIES (EV-Primary) ============
                    elif strategy_type == "Long Straddle":
                        atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
                        atm_put = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
                        
                        if not atm_call.empty and not atm_put.empty:
                            c = atm_call.iloc[0]
                            p = atm_put.iloc[0]
                            
                            exec_cost = c["exec_buy"] + p["exec_buy"]
                            total_slippage = c["slippage"] + p["slippage"]
                            breakeven_up = c["strike"] + exec_cost
                            breakeven_down = p["strike"] - exec_cost
                            move_needed = exec_cost / spot
                            
                            # PROBABILITY CALCULATIONS for straddle
                            # P(profit) = P(spot > BE_up) + P(spot < BE_down)
                            d2_up = (np.log(spot/breakeven_up) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                            d2_down = (np.log(spot/breakeven_down) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                            prob_profit = norm.cdf(d2_up) + (1 - norm.cdf(d2_down))
                            
                            # EV: Expected absolute move minus cost
                            expected_move = spot * mkt_iv_calc * np.sqrt(T) * 0.8  # Expected absolute move
                            ev = expected_move - exec_cost
                            ev_per_risk = ev / exec_cost if exec_cost > 0 else 0
                            
                            if total_slippage > exec_cost * 0.15:
                                rejection_reasons.append(f"Straddle slippage too high: ${total_slippage:.2f} on ${exec_cost:.2f} cost")
                            elif ev <= 0:
                                rejection_reasons.append(f"Straddle EV negative: ${ev:.2f} (need {move_needed:.1%} move, expected {mkt_iv_calc*np.sqrt(T)*0.8:.1%})")
                            else:
                                legs = [
                                    {"type": "CALL", "strike": c["strike"], "direction": "LONG",
                                     "bid": c["bid"], "ask": c["ask"], "exec_price": c["exec_buy"]},
                                    {"type": "PUT", "strike": p["strike"], "direction": "LONG",
                                     "bid": p["bid"], "ask": p["ask"], "exec_price": p["exec_buy"]}
                                ]
                                
                                trade_metrics["ev_dollars"] = ev
                                trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                trade_metrics["prob_profit"] = prob_profit
                                trade_metrics["prob_breakeven"] = prob_profit
                                trade_metrics["liquidity_score"] = 1 - max(c["spread_pct"], p["spread_pct"])
                                trade_metrics["max_loss"] = exec_cost * 100
                                
                                trade_viable = True
                    
                    elif strategy_type == "Long Strangle":
                        otm_calls = calls[calls["strike"] > spot * 1.03]
                        otm_puts = puts[puts["strike"] < spot * 0.97]
                        
                        if not otm_calls.empty and not otm_puts.empty:
                            c = otm_calls.iloc[0]
                            p = otm_puts.iloc[-1]
                            
                            exec_cost = c["exec_buy"] + p["exec_buy"]
                            total_slippage = c["slippage"] + p["slippage"]
                            breakeven_up = c["strike"] + exec_cost
                            breakeven_down = p["strike"] - exec_cost
                            
                            # PROBABILITY CALCULATIONS
                            d2_up = (np.log(spot/breakeven_up) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                            d2_down = (np.log(spot/breakeven_down) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                            prob_profit = norm.cdf(d2_up) + (1 - norm.cdf(d2_down))
                            
                            expected_move = spot * mkt_iv_calc * np.sqrt(T) * 0.8
                            ev = expected_move - exec_cost
                            ev_per_risk = ev / exec_cost if exec_cost > 0 else 0
                            
                            if total_slippage > exec_cost * 0.20:
                                rejection_reasons.append(f"Strangle slippage too high: ${total_slippage:.2f} on ${exec_cost:.2f} cost")
                            elif ev <= 0:
                                rejection_reasons.append(f"Strangle EV negative: ${ev:.2f}")
                            else:
                                legs = [
                                    {"type": "CALL", "strike": c["strike"], "direction": "LONG",
                                     "bid": c["bid"], "ask": c["ask"], "exec_price": c["exec_buy"]},
                                    {"type": "PUT", "strike": p["strike"], "direction": "LONG",
                                     "bid": p["bid"], "ask": p["ask"], "exec_price": p["exec_buy"]}
                                ]
                                
                                trade_metrics["ev_dollars"] = ev
                                trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                trade_metrics["prob_profit"] = prob_profit
                                trade_metrics["prob_breakeven"] = prob_profit
                                trade_metrics["liquidity_score"] = 1 - max(c["spread_pct"], p["spread_pct"])
                                trade_metrics["max_loss"] = exec_cost * 100
                                
                                trade_viable = True
                    
                    elif strategy_type == "Iron Condor":
                        put_short = puts[puts["strike"] < spot * 0.95].iloc[-1] if len(puts[puts["strike"] < spot * 0.95]) > 0 else None
                        put_long = puts[puts["strike"] < spot * 0.90].iloc[-1] if len(puts[puts["strike"] < spot * 0.90]) > 0 else None
                        call_short = calls[calls["strike"] > spot * 1.05].iloc[0] if len(calls[calls["strike"] > spot * 1.05]) > 0 else None
                        call_long = calls[calls["strike"] > spot * 1.10].iloc[0] if len(calls[calls["strike"] > spot * 1.10]) > 0 else None
                        
                        if all([put_short is not None, put_long is not None, call_short is not None, call_long is not None]):
                            exec_credit = (put_short["exec_sell"] - put_long["exec_buy"]) + (call_short["exec_sell"] - call_long["exec_buy"])
                            total_slippage = put_short["slippage"] + put_long["slippage"] + call_short["slippage"] + call_long["slippage"]
                            width = max(put_short["strike"] - put_long["strike"], call_long["strike"] - call_short["strike"])
                            max_loss = width - exec_credit
                            max_profit = exec_credit
                            
                            # PROBABILITY CALCULATIONS
                            # P(max profit) = P(put_short < spot < call_short)
                            d2_put = (np.log(spot/put_short["strike"]) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                            d2_call = (np.log(spot/call_short["strike"]) + (rf/100 - 0.5*mkt_iv_calc**2)*T) / (mkt_iv_calc*np.sqrt(T))
                            prob_max_profit = norm.cdf(d2_call) - (1 - norm.cdf(d2_put))
                            prob_max_profit = max(0, min(1, prob_max_profit))
                            
                            # P(any loss) = 1 - prob_max_profit (simplified)
                            prob_max_loss = 0.1  # Tail estimate
                            
                            # EV CALCULATION
                            ev = (prob_max_profit * max_profit) - ((1 - prob_max_profit) * max_loss * 0.3)  # Partial loss assumption
                            ev_per_risk = ev / max_loss if max_loss > 0 else 0
                            
                            if exec_credit <= 0 or exec_credit < total_slippage:
                                rejection_reasons.append(f"Iron Condor: Credit ${exec_credit:.2f} doesn't cover slippage ${total_slippage:.2f}")
                            elif max_loss <= 0:
                                rejection_reasons.append("Iron Condor: Invalid structure - negative max loss")
                            elif ev <= 0:
                                rejection_reasons.append(f"Iron Condor: EV negative (${ev:.2f})")
                            else:
                                legs = [
                                    {"type": "PUT", "strike": put_long["strike"], "direction": "LONG",
                                     "bid": put_long["bid"], "ask": put_long["ask"], "exec_price": put_long["exec_buy"]},
                                    {"type": "PUT", "strike": put_short["strike"], "direction": "SHORT",
                                     "bid": put_short["bid"], "ask": put_short["ask"], "exec_price": put_short["exec_sell"]},
                                    {"type": "CALL", "strike": call_short["strike"], "direction": "SHORT",
                                     "bid": call_short["bid"], "ask": call_short["ask"], "exec_price": call_short["exec_sell"]},
                                    {"type": "CALL", "strike": call_long["strike"], "direction": "LONG",
                                     "bid": call_long["bid"], "ask": call_long["ask"], "exec_price": call_long["exec_buy"]}
                                ]
                                
                                trade_metrics["ev_dollars"] = ev
                                trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                trade_metrics["prob_profit"] = prob_max_profit
                                trade_metrics["prob_max_profit"] = prob_max_profit
                                trade_metrics["liquidity_score"] = 1 - max(put_short["spread_pct"], call_short["spread_pct"])
                                trade_metrics["max_profit"] = max_profit * 100
                                trade_metrics["max_loss"] = max_loss * 100
                                
                                trade_viable = True
                        else:
                            rejection_reasons.append("Iron Condor: Not enough liquid strikes available")
                    
                    elif strategy_type == "Iron Butterfly":
                        atm = atm_strike
                        atm_call = calls[calls["strike"] == atm].iloc[0] if len(calls[calls["strike"] == atm]) > 0 else None
                        atm_put = puts[puts["strike"] == atm].iloc[0] if len(puts[puts["strike"] == atm]) > 0 else None
                        otm_put = puts[puts["strike"] < atm].iloc[-1] if len(puts[puts["strike"] < atm]) > 0 else None
                        otm_call = calls[calls["strike"] > atm].iloc[0] if len(calls[calls["strike"] > atm]) > 0 else None
                        
                        if all([atm_call is not None, atm_put is not None, otm_put is not None, otm_call is not None]):
                            exec_credit = (atm_call["exec_sell"] + atm_put["exec_sell"]) - (otm_put["exec_buy"] + otm_call["exec_buy"])
                            total_slippage = atm_call["slippage"] + atm_put["slippage"] + otm_put["slippage"] + otm_call["slippage"]
                            width = atm - otm_put["strike"]
                            max_loss = width - exec_credit
                            max_profit = exec_credit
                            
                            # PROBABILITY: Need spot to pin near ATM
                            # P(max profit) is very low for butterflies
                            prob_max_profit = 0.05  # Pin probability is low
                            prob_profit = 0.35  # Rough estimate
                            
                            # EV for butterfly (typically negative due to pin requirement)
                            ev = (prob_max_profit * max_profit) + (prob_profit * max_profit * 0.3) - ((1 - prob_profit) * max_loss * 0.5)
                            ev_per_risk = ev / max_loss if max_loss > 0 else 0
                            
                            if exec_credit <= 0 or exec_credit < total_slippage:
                                rejection_reasons.append(f"Iron Butterfly: Credit ${exec_credit:.2f} doesn't cover slippage ${total_slippage:.2f}")
                            elif ev <= 0:
                                rejection_reasons.append(f"Iron Butterfly: EV negative (${ev:.2f}) - pin probability too low")
                            else:
                                legs = [
                                    {"type": "PUT", "strike": otm_put["strike"], "direction": "LONG",
                                     "bid": otm_put["bid"], "ask": otm_put["ask"], "exec_price": otm_put["exec_buy"]},
                                    {"type": "PUT", "strike": atm, "direction": "SHORT",
                                     "bid": atm_put["bid"], "ask": atm_put["ask"], "exec_price": atm_put["exec_sell"]},
                                    {"type": "CALL", "strike": atm, "direction": "SHORT",
                                     "bid": atm_call["bid"], "ask": atm_call["ask"], "exec_price": atm_call["exec_sell"]},
                                    {"type": "CALL", "strike": otm_call["strike"], "direction": "LONG",
                                     "bid": otm_call["bid"], "ask": otm_call["ask"], "exec_price": otm_call["exec_buy"]}
                                ]
                                
                                trade_metrics["ev_dollars"] = ev
                                trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                trade_metrics["prob_profit"] = prob_profit
                                trade_metrics["prob_max_profit"] = prob_max_profit
                                trade_metrics["liquidity_score"] = 1 - max(atm_call["spread_pct"], atm_put["spread_pct"])
                                trade_metrics["max_profit"] = max_profit * 100
                                trade_metrics["max_loss"] = max_loss * 100
                                
                                trade_viable = True
                        else:
                            rejection_reasons.append("Iron Butterfly: Not enough liquid strikes available")
                    
                    # ============ CALENDAR SPREAD (Trader-Mode Complex Structure) ============
                    elif strategy_type == "Calendar Spread":
                        trader_mode_active = trader_mode == "Trader (Convexity)"
                        calendar_candidates = []
                        
                        # Get available expirations
                        all_exps = hydra.get_expirations(ticker)
                        if len(all_exps) >= 2:
                            front_exp = sel_exp
                            front_days = (datetime.strptime(front_exp, "%Y-%m-%d").date() - date.today()).days
                            
                            # Find back-month expiration (30-60 days further out)
                            back_exps = [e for e in all_exps if e > front_exp]
                            suitable_backs = []
                            for be in back_exps[:5]:
                                be_days = (datetime.strptime(be, "%Y-%m-%d").date() - date.today()).days
                                if 20 <= (be_days - front_days) <= 90:
                                    suitable_backs.append((be, be_days))
                            
                            if suitable_backs:
                                back_exp, back_days = suitable_backs[0]
                                back_chain, _ = hydra.get_chain(ticker, back_exp, spot)
                                
                                if back_chain is not None and not back_chain.empty:
                                    # Add spread calculations to back chain
                                    back_chain["spread"] = back_chain["ask"] - back_chain["bid"]
                                    back_chain["spread_pct"] = np.where(back_chain["bid"] > 0, back_chain["spread"] / back_chain["bid"], 1.0)
                                    
                                    # Filter back chain for liquidity
                                    back_liquid = back_chain[
                                        (back_chain["spread_pct"] <= max_spread_pct) &
                                        (back_chain["bid"] > 0.01)
                                    ].copy()
                                    
                                    if not back_liquid.empty:
                                        back_liquid["exec_buy"] = back_liquid["ask"]
                                        back_liquid["exec_sell"] = back_liquid["bid"]
                                        back_liquid["slippage"] = back_liquid["spread"] / 2
                                        
                                        # Get ATM strike
                                        atm = atm_strike
                                        
                                        # Calendar: same strike, different expiries
                                        front_opt = calls[calls["strike"] == atm] if direction in ["Bullish", "Neutral", "High Volatility"] else puts[puts["strike"] == atm]
                                        back_calls = back_liquid[back_liquid["type"] == "call"]
                                        back_puts = back_liquid[back_liquid["type"] == "put"]
                                        back_opt = back_calls[back_calls["strike"] == atm] if direction in ["Bullish", "Neutral", "High Volatility"] else back_puts[back_puts["strike"] == atm]
                                        
                                        if not front_opt.empty and not back_opt.empty:
                                            front = front_opt.iloc[0]
                                            back = back_opt.iloc[0]
                                            
                                            front_iv = front.get("impliedVolatility", mkt_iv_calc)
                                            back_iv = back.get("impliedVolatility", mkt_iv_calc)
                                            
                                            # Check calendar conditions
                                            cal_ok, cal_reason = check_calendar_conditions(front_iv, back_iv, front_days, back_days)
                                            
                                            # Calendar: Short front, Long back
                                            exec_debit = back["exec_buy"] - front["exec_sell"]
                                            total_slippage = front["slippage"] + back["slippage"]
                                            
                                            if exec_debit > 0 and exec_debit > total_slippage:
                                                # Calendar EV approximation
                                                # Profit from: back vega exposure + front theta decay - debit
                                                front_theta = front.get("theta", -0.02)
                                                back_vega = back.get("vega", 0.1)
                                                
                                                # Expected theta capture (assume front decays ~30%)
                                                theta_gain = abs(front_theta) * front_days * 0.3
                                                # Expected vega gain (if vol rises 5%)
                                                vega_gain = back_vega * 0.05
                                                
                                                ev = (theta_gain + vega_gain - exec_debit) * 100
                                                ev_per_risk = ev / (exec_debit * 100) if exec_debit > 0 else 0
                                                
                                                # Probability of profit: spot stays near ATM
                                                move_threshold = spot * mkt_iv_calc * np.sqrt(front_days/365) * 0.5
                                                d_up = (np.log(spot/(atm + move_threshold))) / (mkt_iv_calc * np.sqrt(front_days/365))
                                                d_down = (np.log(spot/(atm - move_threshold))) / (mkt_iv_calc * np.sqrt(front_days/365))
                                                prob_profit = norm.cdf(d_up) - norm.cdf(d_down)
                                                prob_profit = clamp_probability(prob_profit, front_days/365)
                                                
                                                # Calendar thesis
                                                category, thesis = get_calendar_thesis(front_iv, back_iv, direction)
                                                
                                                if cal_ok:
                                                    thesis = f"{thesis} - {cal_reason}"
                                                
                                                opt_type = "CALL" if direction in ["Bullish", "Neutral", "High Volatility"] else "PUT"
                                                legs = [
                                                    {"type": opt_type, "strike": atm, "direction": "SHORT", "expiry": front_exp,
                                                     "bid": front["bid"], "ask": front["ask"], "exec_price": front["exec_sell"]},
                                                    {"type": opt_type, "strike": atm, "direction": "LONG", "expiry": back_exp,
                                                     "bid": back["bid"], "ask": back["ask"], "exec_price": back["exec_buy"]}
                                                ]
                                                
                                                trade_metrics["ev_dollars"] = ev
                                                trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                                trade_metrics["prob_profit"] = prob_profit
                                                trade_metrics["prob_max_profit"] = prob_profit * 0.7
                                                trade_metrics["prob_confidence"] = 0.6  # Calendar probabilities less certain
                                                trade_metrics["thesis_category"] = category
                                                trade_metrics["thesis"] = thesis
                                                trade_metrics["convexity_score"] = 0.7  # Calendars have term-structure convexity
                                                trade_metrics["trade_label"] = f"🎯 {category.upper()}"
                                                trade_metrics["trade_sublabel"] = "Term-structure convexity play"
                                                liq_score = 1 - max(front["spread_pct"], back["spread_pct"])
                                                trade_metrics["liquidity_score"] = liq_score
                                                trade_metrics["max_loss"] = exec_debit * 100
                                                trade_metrics["complexity"] = 3
                                                trade_metrics["quality_score"] = calc_trade_quality_score(ev, exec_debit * 100, prob_profit, 0.6, liq_score, 3)
                                                
                                                trade_viable = ev > 0
                                                if not trade_viable:
                                                    rejection_reasons.append(f"Calendar EV negative: ${ev:.2f}")
                                            else:
                                                rejection_reasons.append("Calendar: Debit less than slippage")
                                        else:
                                            rejection_reasons.append(f"Calendar: No liquid ATM options at {atm}")
                                    else:
                                        rejection_reasons.append("Calendar: Back-month chain illiquid")
                                else:
                                    rejection_reasons.append("Calendar: Could not fetch back-month chain")
                            else:
                                rejection_reasons.append("Calendar: No suitable back-month expiration (need 20-90 days spread)")
                        else:
                            rejection_reasons.append("Calendar: Need at least 2 expirations available")
                    
                    # ============ DIAGONAL SPREAD (Trader-Mode Complex Structure) ============
                    elif strategy_type == "Diagonal Spread":
                        trader_mode_active = trader_mode == "Trader (Convexity)"
                        
                        all_exps = hydra.get_expirations(ticker)
                        if len(all_exps) >= 2:
                            front_exp = sel_exp
                            front_days = (datetime.strptime(front_exp, "%Y-%m-%d").date() - date.today()).days
                            
                            back_exps = [e for e in all_exps if e > front_exp]
                            suitable_backs = []
                            for be in back_exps[:5]:
                                be_days = (datetime.strptime(be, "%Y-%m-%d").date() - date.today()).days
                                if 20 <= (be_days - front_days) <= 90:
                                    suitable_backs.append((be, be_days))
                            
                            if suitable_backs:
                                back_exp, back_days = suitable_backs[0]
                                back_chain, _ = hydra.get_chain(ticker, back_exp, spot)
                                
                                if back_chain is not None and not back_chain.empty:
                                    # Add spread calculations to back chain
                                    back_chain["spread"] = back_chain["ask"] - back_chain["bid"]
                                    back_chain["spread_pct"] = np.where(back_chain["bid"] > 0, back_chain["spread"] / back_chain["bid"], 1.0)
                                    
                                    back_liquid = back_chain[
                                        (back_chain["spread_pct"] <= max_spread_pct) &
                                        (back_chain["bid"] > 0.01)
                                    ].copy()
                                    
                                    if not back_liquid.empty:
                                        back_liquid["exec_buy"] = back_liquid["ask"]
                                        back_liquid["exec_sell"] = back_liquid["bid"]
                                        back_liquid["slippage"] = back_liquid["spread"] / 2
                                        
                                        atm = atm_strike
                                        
                                        if direction in ["Bullish", "High Volatility"]:
                                            # Bullish diagonal: Short near-term ATM call, Long back-month OTM call
                                            front_opt = calls[calls["strike"] == atm]
                                            back_calls = back_liquid[back_liquid["type"] == "call"]
                                            back_otm = back_calls[back_calls["strike"] > atm * 1.02]
                                            
                                            if not front_opt.empty and not back_otm.empty:
                                                front = front_opt.iloc[0]
                                                back = back_otm.iloc[0]
                                                
                                                exec_debit = back["exec_buy"] - front["exec_sell"]
                                                total_slippage = front["slippage"] + back["slippage"]
                                                
                                                if exec_debit > 0 and exec_debit > total_slippage:
                                                    front_iv = front.get("impliedVolatility", mkt_iv_calc)
                                                    back_iv = back.get("impliedVolatility", mkt_iv_calc)
                                                    
                                                    # Diagonal EV: directional + calendar component
                                                    front_theta = front.get("theta", -0.02)
                                                    back_delta = back.get("delta", 0.4)
                                                    back_vega = back.get("vega", 0.1)
                                                    
                                                    # Expected move contribution
                                                    expected_move = spot * mkt_iv_calc * np.sqrt(back_days/365) * 0.3
                                                    delta_gain = back_delta * expected_move
                                                    theta_gain = abs(front_theta) * front_days * 0.25
                                                    
                                                    ev = (delta_gain + theta_gain - exec_debit) * 100
                                                    ev_per_risk = ev / (exec_debit * 100) if exec_debit > 0 else 0
                                                    
                                                    # Probability: favorable move in underlying
                                                    d_target = (np.log(spot/(back["strike"] - exec_debit)) + (rf/100)*back_days/365) / (mkt_iv_calc * np.sqrt(back_days/365))
                                                    prob_profit = norm.cdf(d_target)
                                                    prob_profit = clamp_probability(prob_profit, back_days/365)
                                                    
                                                    legs = [
                                                        {"type": "CALL", "strike": atm, "direction": "SHORT", "expiry": front_exp,
                                                         "bid": front["bid"], "ask": front["ask"], "exec_price": front["exec_sell"]},
                                                        {"type": "CALL", "strike": back["strike"], "direction": "LONG", "expiry": back_exp,
                                                         "bid": back["bid"], "ask": back["ask"], "exec_price": back["exec_buy"]}
                                                    ]
                                                    
                                                    trade_metrics["ev_dollars"] = ev
                                                    trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                                    trade_metrics["prob_profit"] = prob_profit
                                                    trade_metrics["prob_max_profit"] = prob_profit * 0.6
                                                    trade_metrics["prob_confidence"] = 0.55
                                                    trade_metrics["thesis_category"] = "Hybrid"
                                                    trade_metrics["thesis"] = f"Bullish bias with volatility leverage via time-structure convexity"
                                                    trade_metrics["convexity_score"] = 0.65
                                                    trade_metrics["trade_label"] = "🎯 DIRECTIONAL + VOL"
                                                    trade_metrics["trade_sublabel"] = "Diagonal with upside convexity"
                                                    liq_score = 1 - max(front["spread_pct"], back["spread_pct"])
                                                    trade_metrics["liquidity_score"] = liq_score
                                                    trade_metrics["max_loss"] = exec_debit * 100
                                                    trade_metrics["complexity"] = 3
                                                    trade_metrics["quality_score"] = calc_trade_quality_score(ev, exec_debit * 100, prob_profit, 0.55, liq_score, 3)
                                                    
                                                    trade_viable = ev > 0
                                                    if not trade_viable:
                                                        rejection_reasons.append(f"Diagonal EV negative: ${ev:.2f}")
                                                else:
                                                    rejection_reasons.append("Diagonal: Debit less than slippage")
                                            else:
                                                rejection_reasons.append("Diagonal: No liquid OTM back-month options")
                                        
                                        elif direction in ["Bearish"]:
                                            # Bearish diagonal: Short near-term ATM put, Long back-month OTM put
                                            front_opt = puts[puts["strike"] == atm]
                                            back_puts = back_liquid[back_liquid["type"] == "put"]
                                            back_otm = back_puts[back_puts["strike"] < atm * 0.98]
                                            
                                            if not front_opt.empty and not back_otm.empty:
                                                front = front_opt.iloc[0]
                                                back = back_otm.iloc[-1]
                                                
                                                exec_debit = back["exec_buy"] - front["exec_sell"]
                                                total_slippage = front["slippage"] + back["slippage"]
                                                
                                                if exec_debit > 0 and exec_debit > total_slippage:
                                                    front_theta = front.get("theta", -0.02)
                                                    back_delta = back.get("delta", -0.4)
                                                    
                                                    expected_move = spot * mkt_iv_calc * np.sqrt(back_days/365) * 0.3
                                                    delta_gain = abs(back_delta) * expected_move
                                                    theta_gain = abs(front_theta) * front_days * 0.25
                                                    
                                                    ev = (delta_gain + theta_gain - exec_debit) * 100
                                                    ev_per_risk = ev / (exec_debit * 100) if exec_debit > 0 else 0
                                                    
                                                    d_target = (np.log(spot/(back["strike"] + exec_debit)) + (rf/100)*back_days/365) / (mkt_iv_calc * np.sqrt(back_days/365))
                                                    prob_profit = 1 - norm.cdf(d_target)
                                                    prob_profit = clamp_probability(prob_profit, back_days/365)
                                                    
                                                    legs = [
                                                        {"type": "PUT", "strike": atm, "direction": "SHORT", "expiry": front_exp,
                                                         "bid": front["bid"], "ask": front["ask"], "exec_price": front["exec_sell"]},
                                                        {"type": "PUT", "strike": back["strike"], "direction": "LONG", "expiry": back_exp,
                                                         "bid": back["bid"], "ask": back["ask"], "exec_price": back["exec_buy"]}
                                                    ]
                                                    
                                                    trade_metrics["ev_dollars"] = ev
                                                    trade_metrics["ev_per_dollar_risked"] = ev_per_risk
                                                    trade_metrics["prob_profit"] = prob_profit
                                                    trade_metrics["prob_max_profit"] = prob_profit * 0.6
                                                    trade_metrics["prob_confidence"] = 0.55
                                                    trade_metrics["thesis_category"] = "Hybrid"
                                                    trade_metrics["thesis"] = f"Bearish bias with volatility leverage via time-structure convexity"
                                                    trade_metrics["convexity_score"] = 0.65
                                                    trade_metrics["trade_label"] = "🎯 DIRECTIONAL + VOL"
                                                    trade_metrics["trade_sublabel"] = "Diagonal with downside convexity"
                                                    liq_score = 1 - max(front["spread_pct"], back["spread_pct"])
                                                    trade_metrics["liquidity_score"] = liq_score
                                                    trade_metrics["max_loss"] = exec_debit * 100
                                                    trade_metrics["complexity"] = 3
                                                    trade_metrics["quality_score"] = calc_trade_quality_score(ev, exec_debit * 100, prob_profit, 0.55, liq_score, 3)
                                                    
                                                    trade_viable = ev > 0
                                                    if not trade_viable:
                                                        rejection_reasons.append(f"Diagonal EV negative: ${ev:.2f}")
                                                else:
                                                    rejection_reasons.append("Diagonal: Debit less than slippage")
                                            else:
                                                rejection_reasons.append("Diagonal: No liquid OTM back-month puts")
                                        else:
                                            rejection_reasons.append("Diagonal: Select Bullish or Bearish direction")
                                    else:
                                        rejection_reasons.append("Diagonal: Back-month chain illiquid")
                                else:
                                    rejection_reasons.append("Diagonal: Could not fetch back-month chain")
                            else:
                                rejection_reasons.append("Diagonal: No suitable back-month expiration")
                        else:
                            rejection_reasons.append("Diagonal: Need at least 2 expirations available")
                    
                    # ============ DISPLAY RESULTS (Capital Allocator Output) ============
                    if legs and trade_viable and trade_metrics.get("ev_dollars") is not None and trade_metrics.get("ev_dollars", 0) > 0:
                        ev = trade_metrics.get("ev_dollars", 0) or 0
                        ev_per_risk = trade_metrics.get("ev_per_dollar_risked", 0) or 0
                        prob_profit = trade_metrics.get("prob_profit", 0) or 0
                        prob_max_profit = trade_metrics.get("prob_max_profit") or prob_profit or 0
                        prob_max_loss = trade_metrics.get("prob_max_loss") or 0
                        prob_conf = trade_metrics.get("prob_confidence") or 0.7
                        tail_eps = trade_metrics.get("tail_epsilon") or 0.01
                        is_deep_itm = trade_metrics.get("is_deep_itm", False) or False
                        quality = trade_metrics.get("quality_score") or 0.5
                        trade_label = trade_metrics.get("trade_label") or "✅ ACCEPTABLE"
                        trade_sublabel = trade_metrics.get("trade_sublabel") or ""
                        complexity = trade_metrics.get("complexity", 1)
                        
                        # TRADE QUALITY HEADER
                        label_color = "#10B981" if "🎯" in trade_label or "✅" in trade_label or "📈" in trade_label else "#F59E0B" if "⚡" in trade_label else "#EF4444"
                        st.markdown(f"""<div style="background: linear-gradient(135deg, {label_color}20, {label_color}10); padding: 16px; border-radius: 12px; border: 2px solid {label_color}; margin: 12px 0;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div>
<span style="font-size: 1.3rem; font-weight: 700; color: {label_color};">{trade_label}</span>
<span style="font-size: 0.85rem; color: #94A3B8; margin-left: 12px;">{trade_sublabel}</span>
</div>
<div style="text-align: right;">
<div style="font-size: 0.7rem; color: #64748B;">Quality Score</div>
<div style="font-size: 1.1rem; font-weight: 600; color: {label_color};">{quality:.3f}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
                        
                        # PRIMARY METRICS CARD (EV is dominant)
                        ev_color = "#10B981" if ev > 0.5 else "#F59E0B" if ev > 0 else "#EF4444"
                        conf_color = "#10B981" if prob_conf > 0.7 else "#F59E0B" if prob_conf > 0.5 else "#EF4444"
                        
                        st.markdown(f"""<div style="background: linear-gradient(135deg, #1E293B, #334155); color: white; padding: 20px; border-radius: 12px; margin: 16px 0;">
<h3 style="margin: 0 0 12px 0; color: #10B981;">📋 {strategy_type}</h3>
<p style="color: #94A3B8; margin: 0 0 16px 0;">Underlying: <strong>{ticker}</strong> @ ${spot:.2f} | Expiry: {sel_exp} | Complexity: {complexity}</p>
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
<div style="background: rgba(16,185,129,0.15); padding: 12px; border-radius: 8px; text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase;">Expected Value</div>
<div style="font-size: 1.4rem; font-weight: 700; color: {ev_color};">${ev:.2f}</div>
<div style="font-size: 0.65rem; color: #64748B;">Slippage-adjusted</div>
</div>
<div style="background: rgba(59,130,246,0.15); padding: 12px; border-radius: 8px; text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase;">EV per $ Risked</div>
<div style="font-size: 1.4rem; font-weight: 700; color: #3B82F6;">{ev_per_risk:.1%}</div>
<div style="font-size: 0.65rem; color: #64748B;">Edge metric</div>
</div>
<div style="background: rgba(168,85,247,0.15); padding: 12px; border-radius: 8px; text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase;">P(Profit)</div>
<div style="font-size: 1.4rem; font-weight: 700; color: #A855F7;">{prob_profit:.0%}</div>
<div style="font-size: 0.65rem; color: #64748B;">Above breakeven</div>
</div>
<div style="background: rgba(251,191,36,0.15); padding: 12px; border-radius: 8px; text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase;">Prob Confidence</div>
<div style="font-size: 1.4rem; font-weight: 700; color: {conf_color};">{prob_conf:.0%}</div>
<div style="font-size: 0.65rem; color: #64748B;">Estimate reliability</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
                        
                        # VERTICAL SPREAD: Show P(Max Profit) vs P(Profit) distinction
                        if strategy_type in ["Bull Call Spread", "Bear Put Spread", "Bull Put Spread (Credit)", "Bear Call Spread (Credit)"]:
                            pmax_color = "#10B981" if prob_max_profit > 0.7 else "#F59E0B" if prob_max_profit > 0.4 else "#EF4444"
                            ploss_color = "#EF4444" if prob_max_loss > 0.3 else "#F59E0B" if prob_max_loss > 0.15 else "#10B981"
                            
                            # Warning for deep ITM
                            deep_itm_warning = ""
                            if is_deep_itm:
                                deep_itm_warning = '<div style="background: rgba(239,68,68,0.2); padding: 8px; border-radius: 6px; margin-top: 12px; font-size: 0.8rem; color: #FCA5A5;">⚠️ Deep ITM structure: High probability but concentrated tail risk. Not riskless.</div>'
                            
                            st.markdown(f"""<div style="background: rgba(30,41,59,0.5); padding: 16px; border-radius: 10px; margin: 12px 0;">
<div style="font-size: 0.8rem; color: #64748B; margin-bottom: 10px; text-transform: uppercase;">Vertical Spread Probabilities (Model-Estimated)</div>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
<div style="text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8;">P(Max Profit)</div>
<div style="font-size: 1.2rem; font-weight: 600; color: {pmax_color};">{prob_max_profit:.0%}</div>
<div style="font-size: 0.6rem; color: #64748B;">Spot beyond short strike</div>
</div>
<div style="text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8;">P(Any Profit)</div>
<div style="font-size: 1.2rem; font-weight: 600; color: #A855F7;">{prob_profit:.0%}</div>
<div style="font-size: 0.6rem; color: #64748B;">Above breakeven</div>
</div>
<div style="text-align: center;">
<div style="font-size: 0.7rem; color: #94A3B8;">P(Max Loss)</div>
<div style="font-size: 1.2rem; font-weight: 600; color: {ploss_color};">{prob_max_loss:.0%}</div>
<div style="font-size: 0.6rem; color: #64748B;">Spot beyond long strike</div>
</div>
</div>
<div style="font-size: 0.65rem; color: #64748B; margin-top: 10px; text-align: center;">Tail risk floor applied: {tail_eps:.1%} minimum adverse probability</div>
{deep_itm_warning}
</div>""", unsafe_allow_html=True)
                        
                        # TRADER-FOCUSED THESIS & CONVEXITY DISPLAY
                        thesis_cat = trade_metrics.get("thesis_category", "")
                        thesis = trade_metrics.get("thesis", "")
                        convexity = trade_metrics.get("convexity_score", 0)
                        payoff_accel = trade_metrics.get("payoff_accel", 0)
                        is_eq_sub = trade_metrics.get("is_eq_sub", False)
                        
                        if thesis:
                            conv_color = "#10B981" if convexity > 0.4 else "#F59E0B" if convexity > 0.2 else "#EF4444"
                            accel_color = "#10B981" if payoff_accel > 0.5 else "#F59E0B" if payoff_accel > 0.3 else "#94A3B8"
                            
                            eq_sub_warning = ""
                            if is_eq_sub:
                                eq_sub_warning = '<div style="background: rgba(239,68,68,0.2); padding: 8px; border-radius: 6px; margin-top: 10px; font-size: 0.75rem; color: #FCA5A5;">⚠️ Equity substitute detected: Near-linear payoff, low gamma/vega. Consider ATM alternatives for convexity.</div>'
                            
                            st.markdown(f"""<div style="background: rgba(59,130,246,0.1); padding: 14px; border-radius: 10px; margin: 12px 0; border-left: 3px solid #3B82F6;">
<div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">Trade Thesis</div>
<div style="font-size: 0.95rem; color: #E2E8F0; font-weight: 500;">{thesis}</div>
<div style="display: flex; gap: 20px; margin-top: 12px;">
<div>
<span style="font-size: 0.7rem; color: #94A3B8;">Category:</span>
<span style="font-size: 0.85rem; color: #60A5FA; margin-left: 6px;">{thesis_cat}</span>
</div>
<div>
<span style="font-size: 0.7rem; color: #94A3B8;">Convexity:</span>
<span style="font-size: 0.85rem; color: {conv_color}; margin-left: 6px;">{convexity:.0%}</span>
</div>
<div>
<span style="font-size: 0.7rem; color: #94A3B8;">Payoff Acceleration:</span>
<span style="font-size: 0.85rem; color: {accel_color}; margin-left: 6px;">{payoff_accel:.0%}</span>
</div>
</div>
{eq_sub_warning}
</div>""", unsafe_allow_html=True)
                        
                        # TRADE VIABILITY ASSESSMENT
                        viability_items = []
                        if ev > 0.5:
                            viability_items.append("✅ Strong positive EV")
                        elif ev > 0:
                            viability_items.append("⚡ Marginal positive EV")
                        if prob_profit >= 0.5:
                            viability_items.append("✅ Favorable probability")
                        elif prob_profit >= 0.35:
                            viability_items.append("⚡ Acceptable probability")
                        else:
                            viability_items.append("⚠️ Low win rate")
                        if convexity >= 0.4:
                            viability_items.append("✅ Strong upside convexity")
                        elif convexity >= 0.2:
                            viability_items.append("⚡ Moderate convexity")
                        elif convexity < 0.15:
                            viability_items.append("⚠️ Low convexity - near-linear payoff")
                        if prob_conf >= 0.7:
                            viability_items.append("✅ High confidence estimate")
                        elif prob_conf >= 0.5:
                            viability_items.append("⚡ Moderate confidence")
                        else:
                            viability_items.append("⚠️ Low confidence - verify assumptions")
                        if trade_metrics.get("liquidity_score", 0) >= 0.85:
                            viability_items.append("✅ Excellent liquidity")
                        elif trade_metrics.get("liquidity_score", 0) >= 0.70:
                            viability_items.append("⚡ Adequate liquidity")
                        
                        st.markdown("**📋 Trade Viability Assessment:**")
                        for item in viability_items:
                            st.caption(item)
                        
                        # SECONDARY METRICS
                        sec_col1, sec_col2, sec_col3 = st.columns(3)
                        with sec_col1:
                            if trade_metrics.get("prob_max_profit"):
                                st.metric("P(Max Profit)", f"{trade_metrics['prob_max_profit']:.0%}")
                        with sec_col2:
                            if trade_metrics.get("liquidity_score"):
                                st.metric("Liquidity Score", f"{trade_metrics['liquidity_score']:.0%}")
                        with sec_col3:
                            if trade_metrics.get("max_loss"):
                                st.metric("Max Risk", f"${trade_metrics['max_loss']:.0f}", help="Informational")
                        
                        # METRIC DEFINITIONS
                        with st.expander("📐 Metric Definitions & Formulas", expanded=False):
                            st.markdown("""
                            **EV Formulas Used:**
                            - **Long Options:** `EV ≈ (P_ITM × Avg_Payoff × 0.5) − Entry_Cost`
                            - **Debit Spreads:** `EV ≈ (P_Above_Short − P_Below_Long) × Width × 100 − Debit × 100`
                            - **Credit Spreads:** `EV ≈ Credit × P_Profit − Max_Loss × (1 − P_Profit)`
                            
                            **Probability Proxies:**
                            - Delta proxy with tenor/skew penalties
                            - Implied move proxy: `P(K) ≈ 1 − Φ(|K − S| / Implied_Move)`
                            
                            **Quality Score:**
                            `Quality = (Normalized_EV × Prob_Confidence × Liquidity) ÷ Complexity_Penalty`
                            """)
                        
                        # Trade Legs Table
                        legs_df = pd.DataFrame(legs)
                        st.markdown("**Trade Legs (Execution-Adjusted):**")
                        
                        format_dict = {"strike": "${:.0f}"}
                        if "exec_price" in legs_df.columns:
                            format_dict["exec_price"] = "${:.2f}"
                        if "bid" in legs_df.columns:
                            format_dict["bid"] = "${:.2f}"
                        if "ask" in legs_df.columns:
                            format_dict["ask"] = "${:.2f}"
                        
                        st.dataframe(
                            legs_df.style.format(format_dict),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # INFORMATIONAL METRICS (Secondary - clearly labeled)
                        with st.expander("📊 Secondary Metrics (Informational Only - Do Not Optimize)", expanded=False):
                            st.caption("⚠️ These metrics can mislead. Max profit is rarely achieved. R:R without probability is meaningless.")
                            info_cols = st.columns(3)
                            with info_cols[0]:
                                if trade_metrics.get("max_profit"):
                                    st.metric("Max Profit", f"${trade_metrics['max_profit']:.0f}", help="Rarely achieved")
                            with info_cols[1]:
                                if trade_metrics.get("max_loss"):
                                    st.metric("Max Loss", f"${trade_metrics['max_loss']:.0f}")
                            with info_cols[2]:
                                if trade_metrics.get("rr_ratio"):
                                    st.metric("R:R Ratio", f"{trade_metrics['rr_ratio']:.1f}:1", help="Do not optimize on this alone")
                        
                        # Execution Caveats
                        st.markdown("**🔍 Execution Caveats:**")
                        st.caption("""
                        • Prices use ASK for buys, BID for sells (worst-case fills)
                        • EV assumes model probabilities are accurate
                        • IV changes can significantly impact P/L before expiry
                        • Re-check bid-ask spreads before execution
                        """)
                    
                    elif legs and trade_viable and (trade_metrics["ev_dollars"] is None or trade_metrics["ev_dollars"] <= 0):
                        # EV is zero or negative but structure was found
                        st.warning("⚠️ Structure found but EV ≤ $0 - No positive expected value")
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #78350F, #92400E); color: white; padding: 16px; border-radius: 12px; margin: 12px 0;">
                            <h4 style="margin: 0 0 8px 0; color: #FCD34D;">⚠️ Structurally Valid, Strategically Weak</h4>
                            <p style="color: #FDE68A; font-size: 0.85rem; margin: 0;">
                                This structure exists but does not offer positive expected value after execution costs.
                                Max profit and R:R metrics are misleading without positive EV.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif rejection_reasons:
                        # Check if this is an explicit NO TRADE recommendation
                        is_no_trade = any("NO TRADE" in r for r in rejection_reasons)
                        is_trader_reject = any(x in " ".join(rejection_reasons).lower() for x in ["equity substitute", "convexity", "linear payoff"])
                        
                        if is_no_trade or is_trader_reject:
                            # Determine header based on rejection type
                            header = "🛑 NO TRADE MEETS TRADER-GRADE REQUIREMENTS" if is_trader_reject else "🛑 NO TRADE RECOMMENDED"
                            
                            st.markdown(f"""
<div style="background: linear-gradient(135deg, #1E3A5F, #1E40AF); color: white; padding: 20px; border-radius: 12px; margin: 12px 0; border: 2px solid #3B82F6;">
<h3 style="margin: 0 0 12px 0; color: #60A5FA;">{header}</h3>
<p style="color: #93C5FD; font-size: 0.95rem; margin: 0;">
<strong>This is the correct recommendation.</strong><br>
Trader mode requires non-linear payoffs with meaningful convexity.
Equity-substitute and deep-ITM structures are rejected by default.
</p>
</div>""", unsafe_allow_html=True)
                            
                            st.markdown("**Why No Trade:**")
                            for reason in rejection_reasons:
                                clean = reason.replace('NO TRADE RECOMMENDED: ', '').replace('NO TRADE: ', '')
                                st.info(f"📋 {clean}")
                            
                            st.markdown("""
<div style="background: rgba(59,130,246,0.1); padding: 12px; border-radius: 8px; margin: 12px 0;">
<p style="color: #64748B; font-size: 0.85rem; margin: 0;">
<strong>Trader Mode - Rejection Criteria:</strong><br>
• Trades require upside convexity (gamma/vega exposure)<br>
• Equity-substitute structures (linear payoffs) blocked<br>
• At least one strike must be within ±30% of spot<br>
• Deep ITM spreads de-prioritized<br><br>
<em>To allow equity-like structures, switch to "Capital Substitution" mode.</em>
</p>
</div>""", unsafe_allow_html=True)
                        else:
                            # Regular rejection (not explicit NO TRADE)
                            st.error("❌ No trade offers positive expected value under current assumptions")
                            
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #7F1D1D, #991B1B); color: white; padding: 16px; border-radius: 12px; margin: 12px 0;">
                                <h4 style="margin: 0 0 8px 0; color: #FCA5A5;">🚫 All Candidates Rejected</h4>
                                <p style="color: #FECACA; font-size: 0.85rem; margin: 0;">
                                    The system evaluated all possible structures but none passed EV, probability, and quality filters.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("**Rejection Reasons:**")
                            for reason in rejection_reasons:
                                st.caption(f"• {reason}")
                            
                            # Check if trader mode is active for specific suggestions
                            if trader_mode == "Trader (Convexity)":
                                st.info("""
                                **Trader Mode Active - What this means:**
                                - Deep ITM and equity-substitute structures were NOT generated
                                - All candidates must have convex payoff characteristics
                                - At least one strike must be near-ATM
                                
                                **Try:**
                                - Calendar Spread or Diagonal Spread for term-structure convexity
                                - Closer expiration (more gamma)
                                - ATM or slightly OTM strikes for convexity
                                - Switch to "Capital Substitution" mode for deep ITM access
                                """)
                            else:
                                st.info("""
                                **What this means:**
                                - Expected value is ≤ $0 after slippage
                                - Probability-weighted outcomes don't justify the trade
                                - Liquidity/execution costs consume theoretical edge
                                
                                **Try:**
                                - Different expiration (front-month usually more liquid)
                                - Closer to ATM strikes
                                - Different strategy type
                                - Wait for better market conditions
                                """)
                    else:
                        if trader_mode == "Trader (Convexity)":
                            st.warning("""
                            ⚠️ **No trade meets trader-grade convexity and proximity requirements.**
                            
                            Consider:
                            • Adjusting strike distance threshold
                            • Trying Calendar or Diagonal spreads for term convexity
                            • Enabling "Capital Substitution" mode for equity-like structures
                            """)
                        else:
                            st.warning("⚠️ Could not construct strategy. No liquid options meet criteria.")

    # ==========================================
    # MODULE C: SNIPER (Single Stock Deep Dive)
    # ==========================================
    elif mode == "🎯 Sniper":
        st.title("🎯 Sniper: Single Stock Analysis")
        st.caption("Volatility Squeeze + Volume Velocity + Relative Strength + Monte Carlo")

        # Defaults via session state to mirror prior behavior
        if "ticker" not in st.session_state:
            st.session_state["ticker"] = "TSLA"
        if "window" not in st.session_state:
            st.session_state["window"] = 20
        if "std_dev" not in st.session_state:
            st.session_state["std_dev"] = 2.0
        if "thresh" not in st.session_state:
            st.session_state["thresh"] = 0.25

        def reset_defaults():
            st.session_state["ticker"] = "TSLA"
            st.session_state["window"] = 20
            st.session_state["std_dev"] = 2.0
            st.session_state["thresh"] = 0.25

        with st.expander("⚙️ Strategy Configuration", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ticker = st.text_input("Ticker", value=st.session_state["ticker"]).upper()
                st.session_state["ticker"] = ticker
            with c2:
                window = st.slider("Lookback Window", 10, 60, value=st.session_state["window"])
                st.session_state["window"] = window
            with c3:
                std_dev = st.slider("Volatility (Sigma)", 1.5, 3.0, value=st.session_state["std_dev"])
                st.session_state["std_dev"] = std_dev
            with c4:
                squeeze_thresh = st.slider(
                    "Squeeze Threshold", 0.10, 0.60, value=st.session_state["thresh"], step=0.01
                )
                st.session_state["thresh"] = squeeze_thresh
            st.button("🔄 Reset Defaults", on_click=reset_defaults)

        @st.cache_data(ttl=300)
        def get_quant_data(symbol):
            try:
                tickers = f"{symbol} SPY"
                df = yf.download(tickers, period="2y", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    closes = df["Close"]
                    volumes = df["Volume"]
                else:
                    return None, None, None
                return closes[symbol].dropna(), volumes[symbol].dropna(), closes["SPY"].dropna()
            except Exception:
                return None, None, None

        data, volume, spy_data = get_quant_data(ticker)
        if data is None or data.empty:
            st.error("Data Error. Check ticker or connectivity.")
            st.stop()

        # Technical calculations
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        bandwidth = (upper - lower) / sma

        current_price = data.iloc[-1]
        current_bw = bandwidth.iloc[-1]

        trend_bias = "BULLISH" if current_price > sma.iloc[-1] else "BEARISH"
        ema_9 = data.ewm(span=9, adjust=False).mean()
        ema_21 = data.ewm(span=21, adjust=False).mean()
        ema_signal = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"

        # Quant factor engine
        is_squeeze = current_bw < squeeze_thresh
        squeeze_depth = 0
        if is_squeeze:
            squeeze_depth = (squeeze_thresh - current_bw) / squeeze_thresh
            squeeze_depth = min(max(squeeze_depth, 0), 1)

        vol_3d_ma = volume.rolling(3).mean().iloc[-1]
        vol_30_ma = volume.rolling(30).mean().iloc[-1]
        vol_60_ma = volume.rolling(60).mean().iloc[-1]
        vol_90_ma = volume.rolling(90).mean().iloc[-1]

        velocity_score = 0
        if vol_3d_ma > vol_30_ma:
            velocity_score += 1
        if vol_3d_ma > vol_60_ma:
            velocity_score += 1
        if vol_3d_ma > vol_90_ma:
            velocity_score += 1

        if velocity_score >= 2:
            vol_status = "Speeding Up"
            vol_color = "normal"
        elif velocity_score <= 0:
            vol_status = "Slowing Down"
            vol_color = "inverse"
        else:
            vol_status = "Neutral"
            vol_color = "off"

        df_rs = pd.DataFrame({"Stock": data, "SPY": spy_data}).dropna()
        df_rs["RS_Ratio"] = df_rs["Stock"].pct_change(60) - df_rs["SPY"].pct_change(60)
        current_rs = df_rs["RS_Ratio"].iloc[-1]

        returns = df_rs.pct_change().dropna()
        cov = returns["Stock"].cov(returns["SPY"])
        var = returns["SPY"].var()
        beta = cov / var if var != 0 else np.nan

        confidence = 0
        if trend_bias == "BULLISH" and ema_signal == "BULLISH":
            confidence += 35
        elif trend_bias == "BEARISH" and ema_signal == "BEARISH":
            confidence += 35
        else:
            confidence += 15

        if is_squeeze:
            confidence += 15 + (squeeze_depth * 10)

        if vol_status == "Speeding Up":
            confidence += 20
        elif vol_status == "Neutral":
            confidence += 10

        if trend_bias == "BULLISH" and current_rs > 0:
            confidence += 20
        elif trend_bias == "BEARISH" and current_rs < 0:
            confidence += 20

        confidence = min(int(confidence), 99)

        st.subheader("🤖 Quant Model Output")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${current_price:.2f}")
        m2.metric(
            "Trend Bias (SMA)",
            trend_bias,
            delta="Long" if trend_bias == "BULLISH" else "Short",
            delta_color="normal" if trend_bias == "BULLISH" else "inverse",
        )
        m3.metric(
            "Momentum (EMA)",
            ema_signal,
            delta="Strong" if ema_signal == trend_bias else "Weak",
            help="9 EMA vs 21 EMA crossover.",
        )
        m4.metric("Squeeze Status", "COILED" if is_squeeze else "LOOSE", delta=f"BW: {current_bw:.3f}", delta_color="inverse")

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Confidence Factor", f"{confidence}%", help="Weighted score of Trend, Squeeze Depth, Volume Velocity, and RS.")
            st.progress(confidence / 100)
        with q2:
            st.metric(
                "Volume Velocity",
                vol_status,
                delta="vs 30/60/90d Avg",
                delta_color=vol_color,
                help="Compares current volume flow against 30, 60, and 90-day baselines.",
            )
        with q3:
            st.metric("Relative Strength", f"{current_rs:.1%}", delta="vs SPY (60d)", help="Performance differential vs SPY over last 60 days.")
        with q4:
            st.metric("Risk (Beta)", f"{beta:.2f}", help="Volatility relative to S&P 500. >1.0 is aggressive, <1.0 is defensive.")

        st.write("")
        tab1, tab2 = st.tabs(["💰 Price Action", "📉 Bandwidth Analyzer"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data, mode="lines", name="Price", line=dict(color="#0F172A", width=1.5)))
            fig.add_trace(go.Scatter(x=upper.index, y=upper, mode="lines", name="Upper", line=dict(color="#10B981", width=1, dash="dot")))
            fig.add_trace(
                go.Scatter(
                    x=lower.index,
                    y=lower,
                    mode="lines",
                    name="Lower",
                    line=dict(color="#10B981", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(16, 185, 129, 0.05)",
                )
            )
            fig.add_trace(go.Scatter(x=sma.index, y=sma, mode="lines", name="SMA", line=dict(color="#F59E0B", width=1.5)))
            fig.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            bw_fig = go.Figure()
            bw_fig.add_trace(go.Scatter(x=bandwidth.index[-180:], y=bandwidth.tail(180), mode="lines", name="Bandwidth", line=dict(color="#3B82F6", width=2)))
            bw_fig.add_hline(y=squeeze_thresh, line_dash="dash", line_color="#EF4444", annotation_text=f"Threshold ({squeeze_thresh})")
            bw_fig.update_layout(height=400, template="plotly_white", title="Historical Bandwidth vs Threshold", hovermode="x unified")
            st.plotly_chart(bw_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🎲 Monte Carlo Scenario (30 Days)")
        st.caption("Simulating 1,000 future paths based on historical volatility.")

        daily_vol = np.log(data / data.shift(1)).std()
        annual_vol = daily_vol * np.sqrt(252)

        SIMULATIONS = 1000
        DAYS = 30
        random_shocks = np.random.normal(0, daily_vol, (DAYS, SIMULATIONS))
        price_paths = current_price * (1 + random_shocks).cumprod(axis=0)

        final_prices = price_paths[-1]
        profitability = (final_prices > current_price).mean()

        mc_fig = go.Figure()
        mc_fig.add_trace(
            go.Scatter(
                x=np.tile(np.arange(DAYS), SIMULATIONS),
                y=price_paths.flatten(order="F"),
                mode="lines",
                line=dict(color="#10B981", width=0.5),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        mc_fig.add_trace(go.Scatter(x=np.arange(DAYS), y=price_paths.mean(axis=1), mode="lines", name="Mean Path", line=dict(color="black", width=2)))
        mc_fig.add_hline(y=current_price, line_dash="dash", line_color="black")
        mc_fig.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0))

        c_mc1, c_mc2 = st.columns([3, 1])
        with c_mc1:
            st.plotly_chart(mc_fig, use_container_width=True)
        with c_mc2:
            st.metric("Probability of Profit", f"{profitability:.1%}", help="Percentage of 1000 scenarios that end positive.")
            st.metric("Projected Volatility", f"{annual_vol:.1%}")

        st.markdown(
            """
        <div class="edu-footer">
            <h3>🎓 Model Architecture & Logic</h3>
            <p><strong>1. Volatility Squeeze:</strong> Identifies potential energy. BW < Threshold = Squeeze.</p>
            <p><strong>2. Volume Velocity:</strong> Analyzes speed of participation vs 30/60/90d baselines.</p>
            <p><strong>3. Relative Strength:</strong> Performance differential vs SPY.</p>
            <p><strong>4. Risk Parity:</strong> Beta indicates volatility relative to market.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ==========================================
    # MODULE D: THE HUNTER (Wide Net Scanner: Ryan Model)
    # ==========================================
    elif mode == "🦅 Hunter":
        st.title("🦅 The Hunter: Market Scanner")
        st.caption("Cast the net wide | Ryan Model + Trap Detection + Momentum Scoring")

        with st.expander("🛠️ Scanner Settings", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                scan_window = st.slider("Lookback Window", 10, 50, 20)
            with sc2:
                scan_sqz_thresh = st.slider("Squeeze Threshold", 0.10, 0.40, 0.25)
            with sc3:
                min_confidence = st.slider("Min Confidence %", 50, 95, 80)

        # Single-name stocks only (no ETFs) - High volume, quality tickers
        TICKER_LIST = [
            # Mega-cap Tech
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE", 
            "CRM", "AMD", "QCOM", "TXN", "INTC", "IBM", "MU", "NOW", "UBER", "PANW", "SNOW", "NET", "CRWD", "DDOG", "ZS",
            # Financials
            "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "AXP", "BLK", "C", "PYPL", "HOOD", "COIN", "SOFI", "SCHW", "CME",
            # Consumer
            "WMT", "COST", "PG", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "TGT", "LOW", "TJX", "LULU", "CMG", "YUM",
            # Healthcare
            "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "PFE", "AMGN", "ISRG", "BMY", "GILD", "CVS", "MDT", "DHR", "ABT",
            # Industrial & Energy
            "CAT", "DE", "HON", "GE", "UNP", "UPS", "BA", "LMT", "RTX", "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "HAL",
            # High-Beta / Momentum
            "MSTR", "MARA", "PLTR", "DKNG", "ROKU", "SQ", "AFRM", "RIOT", "CLSK", "CVNA", "UPST", "AI", "GME", "AMC",
            # Additional Quality Names
            "NFLX", "BKNG", "ABNB", "SHOP", "SQ", "SPOT", "SNAP", "PINS", "RBLX", "U", "TTD", "TEAM", "DOCU", "ZM",
        ]

        @st.cache_data(ttl=600)
        def batch_process_tickers(tickers, scan_window, scan_sqz_thresh):
            results = []
            data_batch = yf.download(tickers + ["SPY"], period="6mo", progress=False)

            def get_series(df, symbol, col):
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        return df[col][symbol]
                    else:
                        return df[col]
                except Exception:
                    return None

            for symbol in tickers:
                try:
                    closes = get_series(data_batch, symbol, "Close")
                    highs = get_series(data_batch, symbol, "High")
                    lows = get_series(data_batch, symbol, "Low")
                    spy_closes = get_series(data_batch, "SPY", "Close")

                    if closes is None or highs is None or lows is None or spy_closes is None:
                        continue
                    closes = closes.dropna()
                    highs = highs.dropna()
                    lows = lows.dropna()
                    spy_closes = spy_closes.dropna()

                    if len(closes) < max(scan_window, 60):
                        continue

                    sma = closes.rolling(window=scan_window).mean()
                    std = closes.rolling(window=scan_window).std()
                    upper = sma + (std * 2.0)
                    lower = sma - (std * 2.0)
                    bandwidth = (upper - lower) / sma
                    current_price = closes.iloc[-1]
                    current_bw = bandwidth.iloc[-1]
                    is_squeeze = current_bw < scan_sqz_thresh

                    squeeze_depth = 0
                    if is_squeeze:
                        squeeze_depth = (scan_sqz_thresh - current_bw) / scan_sqz_thresh
                        squeeze_depth = min(max(squeeze_depth, 0), 1)

                    trend_bias = "BULLISH" if current_price > sma.iloc[-1] else "BEARISH"
                    ema_9 = closes.ewm(span=9, adjust=False).mean()
                    ema_21 = closes.ewm(span=21, adjust=False).mean()
                    ema_signal = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"

                    stock_pct = closes.pct_change(60).iloc[-1]
                    spy_pct = spy_closes.pct_change(60).iloc[-1]
                    rs_ratio = stock_pct - spy_pct

                    curr_high = highs.iloc[-1]
                    curr_low = lows.iloc[-1]
                    day_range = curr_high - curr_low
                    range_position = 0.5 if day_range == 0 else (current_price - curr_low) / day_range
                    if range_position > 0.8:
                        intraday_health = "POWER"
                    elif range_position < 0.2:
                        intraday_health = "WEAK"
                    elif range_position < 0.5 and trend_bias == "BULLISH":
                        intraday_health = "FADING"
                    else:
                        intraday_health = "NEUTRAL"

                    confidence = 0
                    if trend_bias == "BULLISH" and ema_signal == "BULLISH":
                        confidence += 35
                    elif trend_bias == "BEARISH" and ema_signal == "BEARISH":
                        confidence += 35
                    else:
                        confidence += 15

                    if is_squeeze:
                        confidence += 15 + (squeeze_depth * 10)

                    if trend_bias == "BULLISH" and rs_ratio > 0:
                        confidence += 20
                    elif trend_bias == "BEARISH" and rs_ratio < 0:
                        confidence += 20

                    if intraday_health in ["FADING", "WEAK"]:
                        confidence -= 15

                    confidence = min(max(int(confidence), 0), 99)

                    action_signal = "WAIT"
                    if confidence >= 80 and intraday_health == "POWER" and trend_bias == "BULLISH":
                        action_signal = "✅ BUY"
                    elif confidence >= 80 and intraday_health == "WEAK" and trend_bias == "BEARISH":
                        action_signal = "🔻 SHORT"

                    results.append(
                        {
                            "Ticker": symbol,
                            "Price": current_price,
                            "Action": action_signal,
                            "Trend": trend_bias,
                            "Momentum": ema_signal,
                            "Squeeze": "COILED" if is_squeeze else "LOOSE",
                            "Bandwidth": current_bw,
                            "Confidence": confidence,
                            "RS_vs_SPY": rs_ratio,
                            "Health": intraday_health,
                        }
                    )
                except Exception:
                    continue
            return pd.DataFrame(results)

        if st.button("🚀 Initialize Scan Sequence"):
            progress_text = "Connecting to Neural Lattice... Please wait."
            my_bar = st.progress(0, text=progress_text)

            with st.spinner(f"Scanning {len(TICKER_LIST)} Assets..."):
                my_bar.progress(30, text="Downloading Batch Data...")
                df_results = batch_process_tickers(TICKER_LIST, scan_window, scan_sqz_thresh)
                my_bar.progress(90, text="Applying Ryan Model Logic...")

                if df_results is not None and not df_results.empty:
                    my_bar.progress(100, text="Scan Complete.")
                    my_bar.empty()

                    longs = df_results[
                        (df_results["Trend"] == "BULLISH")
                        & (df_results["Momentum"] == "BULLISH")
                        & (df_results["Squeeze"] == "COILED")
                        & (df_results["Confidence"] >= min_confidence)
                    ]
                    shorts = df_results[
                        (df_results["Trend"] == "BEARISH")
                        & (df_results["Momentum"] == "BEARISH")
                        & (df_results["Squeeze"] == "COILED")
                        & (df_results["Confidence"] >= min_confidence)
                    ]

                    # Custom styling for buy signals - green gradient by confidence
                    def style_longs(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for idx in df.index:
                            conf = df.loc[idx, "Confidence"]
                            action = df.loc[idx, "Action"]
                            health = df.loc[idx, "Health"]
                            # Intensity based on confidence (darker green = higher confidence)
                            if "BUY" in str(action):
                                # MUST BUY - dark green
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.4); font-weight: bold;"
                            elif conf >= 90 and health == "POWER":
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.35);"
                            elif conf >= 85:
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.25);"
                            elif conf >= 80:
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.15);"
                            else:
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.08);"
                        return styles

                    st.subheader(f"🟢 Long Setups ({len(longs)})")
                    if not longs.empty:
                        longs_sorted = longs.sort_values(by="Confidence", ascending=False)
                        st.dataframe(
                            longs_sorted.style.apply(style_longs, axis=None).format({"Price": "${:.2f}", "Bandwidth": "{:.4f}", "RS_vs_SPY": "{:.2%}"}),
                            use_container_width=True,
                            column_config={
                                "Confidence": st.column_config.ProgressColumn("Confidence Score", format="%d%%", min_value=0, max_value=100),
                                "Health": st.column_config.TextColumn("Intraday State", help="POWER: Closing high. FADING: Up but dropping (Trap Warning)."),
                            },
                            column_order=("Ticker", "Action", "Confidence", "Health", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                            hide_index=True,
                        )
                        # Highlight top picks
                        top_buys = longs_sorted[longs_sorted["Action"].str.contains("BUY")]
                        if not top_buys.empty:
                            st.success(f"🎯 **TOP PICKS:** {', '.join(top_buys['Ticker'].head(5).tolist())}")
                    else:
                        st.info(f"No Long setups found (Conf > {min_confidence}% + Coiled).")

                    st.markdown("---")

                    # Custom styling for short signals - red gradient by confidence
                    def style_shorts(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for idx in df.index:
                            conf = df.loc[idx, "Confidence"]
                            action = df.loc[idx, "Action"]
                            health = df.loc[idx, "Health"]
                            if "SHORT" in str(action):
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.4); font-weight: bold;"
                            elif conf >= 90 and health == "WEAK":
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.35);"
                            elif conf >= 85:
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.25);"
                            elif conf >= 80:
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.15);"
                            else:
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.08);"
                        return styles

                    st.subheader(f"🔴 Short Setups ({len(shorts)})")
                    if not shorts.empty:
                        shorts_sorted = shorts.sort_values(by="Confidence", ascending=False)
                        st.dataframe(
                            shorts_sorted.style.apply(style_shorts, axis=None).format({"Price": "${:.2f}", "Bandwidth": "{:.4f}", "RS_vs_SPY": "{:.2%}"}),
                            use_container_width=True,
                            column_config={"Confidence": st.column_config.ProgressColumn("Confidence Score", format="%d%%", min_value=0, max_value=100)},
                            column_order=("Ticker", "Action", "Confidence", "Health", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                            hide_index=True,
                        )
                        top_shorts = shorts_sorted[shorts_sorted["Action"].str.contains("SHORT")]
                        if not top_shorts.empty:
                            st.error(f"🎯 **TOP SHORTS:** {', '.join(top_shorts['Ticker'].head(5).tolist())}")
                    else:
                        st.info(f"No Short setups found (Conf > {min_confidence}% + Coiled).")

                    with st.expander("📂 View Full Scan Results (All Assets)"):
                        st.dataframe(
                            df_results.sort_values(by="Confidence", ascending=False).style.format({"Price": "${:.2f}", "Confidence": "{:.0f}%"}),
                            use_container_width=True,
                        )

                    st.markdown("---")
                    st.markdown(
                        """
                    ### 🔑 Hunter Signal Decoder

                    **Action Signals (highlighted rows)**
                    | Signal | Meaning |
                    |--------|---------|
                    | ✅ **BUY** | Conf ≥80% + POWER health + Bullish — **ride the momentum** |
                    | 🔻 **SHORT** | Conf ≥80% + WEAK health + Bearish — **fade the weakness** |
                    | WAIT | Setup present but missing conviction |

                    **Health Status (Trap Detection)**
                    | Status | What It Means |
                    |--------|---------------|
                    | 🟢 POWER | Closing near day's high — institutions holding |
                    | 🟡 NEUTRAL | Mid-range chop — wait for clarity |
                    | 🔴 FADING | Bullish but dropping back — trap risk |
                    | 🔴 WEAK | Closing near day's low — bearish confirmation |

                    **Row Colors:** Darker green/red = higher confidence + actionable signal
                    """
                    )
                else:
                    st.error("Scan returned no data. Please check API connection.")
        else:
            st.info("🦅 **Ready to Hunt.** Click to scan 100+ assets using the Ryan Model — momentum, squeeze, trap detection, and relative strength.")

    # ==========================================
    # MODULE E: OPTION HUNTER (Options Scanner)
    # ==========================================
    elif mode == "🔍 Opt Hunt":
        st.title("🔍 Option Hunter: Mispriced Options Scanner")
        st.caption("Scan the market for underpriced options with edge")
        
        # Scanner Settings
        st.markdown("### ⚙️ Scanner Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dte_range = st.selectbox("Days to Expiration", ["7-14 days", "14-30 days", "30-60 days", "60-90 days"], index=1)
        with col2:
            scan_direction = st.selectbox("Direction", ["Bullish (Calls)", "Bearish (Puts)", "Both"])
        with col3:
            min_pop = st.slider("Min PoP %", 30, 80, 50)
        with col4:
            max_spread_pct = st.slider("Max Spread %", 5, 30, 15)
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            min_edge = st.slider("Min Edge %", 0, 30, 5)
        with col6:
            max_otm_pct = st.slider("Max OTM %", 5, 30, 15)
        with col7:
            rf_rate = st.number_input("Risk Free %", value=4.5, step=0.1)
        with col8:
            max_results = st.selectbox("Top Results", [5, 10, 20, 50], index=1)
        
        st.caption("💡 Single-name stocks only. For ETF options, use **📊 ETF Opts** module.")
        
        # No ETFs in this scanner - they have their own module
        INDEX_ETFS = set()  # Empty set - no ETFs
        allow_index_etfs = False
        prefer_single_names = True
        
        # Parse DTE range
        dte_map = {
            "7-14 days": (7, 14),
            "14-30 days": (14, 30),
            "30-60 days": (30, 60),
            "60-90 days": (60, 90)
        }
        min_dte, max_dte = dte_map[dte_range]
        
        # Stock Universe (same as Hunter) - Single names first, then ETFs
        # Single-name stocks only (no ETFs) - High volume, optionable tickers
        OPT_TICKER_LIST = [
            # Mega-cap Tech (highest options liquidity)
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE", 
            "CRM", "AMD", "QCOM", "TXN", "INTC", "IBM", "MU", "NOW", "UBER", "PANW", "SNOW", "NET", "CRWD", "DDOG",
            # Financials
            "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "AXP", "BLK", "C", "PYPL", "COIN", "HOOD", "SCHW",
            # Consumer
            "WMT", "COST", "PG", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "LULU", "CMG",
            # Healthcare
            "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "PFE", "AMGN", "ISRG", "MDT",
            # Industrial & Energy
            "CAT", "DE", "HON", "GE", "UNP", "BA", "LMT", "RTX", "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
            # High-Beta / Momentum (great for options)
            "MSTR", "MARA", "PLTR", "DKNG", "ROKU", "SQ", "RIOT", "CVNA", "GME", "AMC",
            # Additional High-Volume Options Names
            "NFLX", "BKNG", "ABNB", "SHOP", "SPOT", "SNAP", "RBLX", "TTD",
        ]
        
        @st.cache_data(ttl=300, show_spinner=False)
        def scan_options_universe(tickers, min_dte, max_dte, direction, min_pop_thresh, max_spread, min_edge_pct, max_otm, rf, index_etf_set, apply_index_penalty):
            """Scan multiple tickers for underpriced options"""
            results = []
            hydra_scan = HydraEngine()
            
            for ticker in tickers:
                try:
                    # Get spot price
                    spot, _ = hydra_scan.get_spot(ticker)
                    if spot is None or spot <= 0:
                        continue
                    
                    # Get expirations
                    exps = hydra_scan.get_expirations(ticker)
                    if not exps:
                        continue
                    
                    # Find expiration in target range
                    target_exp = None
                    target_dte = None
                    for exp in exps:
                        try:
                            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                            dte = (exp_date - date.today()).days
                            if min_dte <= dte <= max_dte:
                                target_exp = exp
                                target_dte = dte
                                break
                        except:
                            continue
                    
                    if not target_exp:
                        continue
                    
                    # Get options chain
                    chain, _ = hydra_scan.get_chain(ticker, target_exp, spot)
                    if chain is None or chain.empty:
                        continue
                    
                    # Add spread calculations
                    chain["spread"] = chain["ask"] - chain["bid"]
                    chain["spread_pct"] = np.where(chain["bid"] > 0, (chain["spread"] / chain["bid"]) * 100, 100)
                    
                    # Filter by type based on direction
                    if direction == "Bullish (Calls)":
                        chain = chain[chain["type"] == "call"]
                    elif direction == "Bearish (Puts)":
                        chain = chain[chain["type"] == "put"]
                    
                    # Liquidity filter
                    liquid_chain = chain[
                        (chain["spread_pct"] <= max_spread) &
                        (chain["bid"] > 0.05) &
                        ((chain["volume"] >= 10) | (chain["openInterest"] >= 100))
                    ].copy()
                    
                    if liquid_chain.empty:
                        continue
                    
                    # Calculate OTM percentage and filter
                    liquid_chain["otm_pct"] = np.where(
                        liquid_chain["type"] == "call",
                        ((liquid_chain["strike"] - spot) / spot) * 100,
                        ((spot - liquid_chain["strike"]) / spot) * 100
                    )
                    
                    # Filter: slightly OTM to max_otm (no deep ITM)
                    liquid_chain = liquid_chain[(liquid_chain["otm_pct"] >= -5) & (liquid_chain["otm_pct"] <= max_otm)]
                    
                    if liquid_chain.empty:
                        continue
                    
                    # Calculate model prices and edge for remaining options
                    T = target_dte / 365.0
                    
                    for _, row in liquid_chain.iterrows():
                        try:
                            strike = row["strike"]
                            opt_type = row["type"]
                            market_mid = (row["bid"] + row["ask"]) / 2
                            iv = row.get("impliedVolatility", 0.3)
                            if iv <= 0 or iv > 3:
                                iv = 0.3
                            
                            # Calculate model price (BSM)
                            kernel = PricingKernel(spot, strike, T, rf / 100, iv, opt_type)
                            model_price = kernel.price_bsm()
                            
                            # Edge calculation (buy at ASK, so edge = model - ask)
                            ask_price = row["ask"]
                            edge = model_price - ask_price
                            edge_pct = (edge / ask_price * 100) if ask_price > 0 else 0
                            
                            # Skip if not underpriced enough
                            if edge_pct < min_edge_pct:
                                continue
                            
                            # Calculate PoP using delta proxy
                            greeks = kernel.get_greeks()
                            delta = greeks.delta
                            
                            # PoP approximation: for long calls, P(ITM) ~ |delta|
                            # Adjust for direction and account for theta decay
                            if opt_type == "call":
                                pop_raw = abs(delta)
                            else:
                                pop_raw = abs(delta)
                            
                            # Adjust PoP for OTM options (they need larger moves)
                            pop_adjusted = pop_raw * (1 - row["otm_pct"] / 100 * 0.5)
                            pop_adjusted = max(0.05, min(0.95, pop_adjusted))
                            
                            # Skip if below min PoP
                            if pop_adjusted * 100 < min_pop_thresh:
                                continue
                            
                            # Calculate expected value
                            # EV = PoP * avg_profit - (1-PoP) * cost
                            breakeven_move = (strike + ask_price - spot) / spot if opt_type == "call" else (spot - strike + ask_price) / spot
                            potential_profit = model_price * 1.5 - ask_price  # Conservative upside
                            ev = pop_adjusted * potential_profit - (1 - pop_adjusted) * ask_price
                            ev_per_risk = (ev / ask_price) if ask_price > 0 else 0
                            
                            # Quality score (base)
                            quality_base = (edge_pct / 100) * 0.3 + pop_adjusted * 0.4 + (1 - row["spread_pct"] / max_spread) * 0.3
                            
                            # Check if this is an index ETF
                            is_index = ticker in index_etf_set
                            
                            # Apply index ETF penalty if requested (15% penalty to prevent crowding out single names)
                            if is_index and apply_index_penalty:
                                quality = quality_base * 0.85
                            else:
                                quality = quality_base
                            
                            results.append({
                                "Ticker": ticker,
                                "Type": opt_type.upper(),
                                "Strike": strike,
                                "Expiry": target_exp,
                                "DTE": target_dte,
                                "Spot": spot,
                                "OTM%": row["otm_pct"],
                                "Bid": row["bid"],
                                "Ask": row["ask"],
                                "Model": model_price,
                                "Edge$": edge,
                                "Edge%": edge_pct,
                                "PoP%": pop_adjusted * 100,
                                "EV$": ev,
                                "Delta": delta,
                                "IV": iv * 100,
                                "Spread%": row["spread_pct"],
                                "Volume": row["volume"],
                                "OI": row["openInterest"],
                                "Quality": quality,
                                "QualityRaw": quality_base,
                                "IsETF": is_index
                            })
                            
                        except Exception as opt_err:
                            continue
                            
                except Exception as ticker_err:
                    continue
            
            return pd.DataFrame(results)
        
        # Run Scanner
        if st.button("🚀 Scan for Underpriced Options", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing scanner...")
            
            with st.spinner(f"Scanning {len(OPT_TICKER_LIST)} tickers for mispriced options..."):
                progress_bar.progress(20, text="Fetching options chains...")
                
                df_results = scan_options_universe(
                    OPT_TICKER_LIST, 
                    min_dte, 
                    max_dte, 
                    scan_direction, 
                    min_pop, 
                    max_spread_pct, 
                    min_edge, 
                    max_otm_pct,
                    rf_rate,
                    INDEX_ETFS,
                    prefer_single_names
                )
                
                progress_bar.progress(90, text="Ranking opportunities...")
                
                if df_results is not None and not df_results.empty:
                    progress_bar.progress(100, text="Scan complete!")
                    progress_bar.empty()
                    
                    # ============ PER-SYMBOL RANKING (Step 1) ============
                    # For each ticker, keep only the best option (top 1 per symbol)
                    df_results = df_results.sort_values("Quality", ascending=False)
                    best_per_symbol = df_results.groupby("Ticker").head(1).copy()
                    
                    # ============ DIVERSITY ENFORCEMENT (Step 2) ============
                    # Separate single names from ETFs
                    single_name_results = best_per_symbol[~best_per_symbol["IsETF"]]
                    etf_results = best_per_symbol[best_per_symbol["IsETF"]]
                    
                    # Check if we have any single names
                    has_single_names = len(single_name_results) > 0
                    only_etfs = len(single_name_results) == 0 and len(etf_results) > 0
                    
                    # Build diverse final list: prioritize single names, fill with ETFs if needed
                    if has_single_names:
                        # Take top single names first
                        final_results = single_name_results.head(max_results)
                        
                        # If we still have room and ETFs are allowed, add some ETFs
                        remaining_slots = max_results - len(final_results)
                        if remaining_slots > 0 and allow_index_etfs and len(etf_results) > 0:
                            # Add at most 2 ETFs even if more slots available (diversity rule)
                            etf_to_add = min(remaining_slots, 2)
                            final_results = pd.concat([final_results, etf_results.head(etf_to_add)])
                    else:
                        # Only ETFs passed filters
                        final_results = etf_results.head(max_results)
                    
                    # Sort final results by Quality
                    final_results = final_results.sort_values("Quality", ascending=False)
                    
                    # ============ EXPLICIT ETF-ONLY WARNING ============
                    if only_etfs:
                        st.warning("""
                        ⚠️ **Only Index ETF structures currently satisfy your quality filters.**
                        
                        No single-name stocks passed the minimum criteria. Consider:
                        - Lowering Min Edge % or Min PoP %
                        - Increasing Max Spread %
                        - Trying a different DTE range
                        """)
                    
                    # Stats summary
                    st.markdown("---")
                    st.markdown("### 📊 Scan Results")
                    
                    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
                    with stat1:
                        st.metric("Diverse Ideas", len(final_results))
                    with stat2:
                        st.metric("Total Options Scanned", len(df_results))
                    with stat3:
                        avg_edge = final_results["Edge%"].mean() if len(final_results) > 0 else 0
                        st.metric("Avg Edge %", f"{avg_edge:.1f}%")
                    with stat4:
                        avg_pop = final_results["PoP%"].mean() if len(final_results) > 0 else 0
                        st.metric("Avg PoP %", f"{avg_pop:.1f}%")
                    with stat5:
                        single_count = len(final_results[~final_results["IsETF"]]) if len(final_results) > 0 else 0
                        etf_count = len(final_results[final_results["IsETF"]]) if len(final_results) > 0 else 0
                        st.metric("Singles / ETFs", f"{single_count} / {etf_count}")
                    
                    st.markdown("---")
                    
                    # Top Opportunities (Diverse Selection)
                    st.subheader(f"🎯 Top Underpriced Options — Diverse Selection ({len(final_results)} unique tickers)")
                    
                    # Display columns selection
                    display_cols = ["Ticker", "Type", "Strike", "DTE", "Spot", "OTM%", "Ask", "Model", "Edge%", "PoP%", "Delta", "IV", "Quality"]
                    
                    st.dataframe(
                        final_results[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Type": st.column_config.TextColumn("Type", width="small"),
                            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                            "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
                            "OTM%": st.column_config.NumberColumn("OTM%", format="%.1f%%"),
                            "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                            "Model": st.column_config.NumberColumn("Model", format="$%.2f"),
                            "Edge%": st.column_config.NumberColumn("Edge%", format="%.1f%%"),
                            "PoP%": st.column_config.NumberColumn("PoP%", format="%.1f%%"),
                            "Delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                            "IV": st.column_config.NumberColumn("IV", format="%.1f%%"),
                            "Quality": st.column_config.ProgressColumn("Quality", format="%.2f", min_value=0, max_value=1)
                        }
                    )
                    
                    # Detailed view expander - show ALL candidates per symbol
                    with st.expander("📋 Full Results (All Options, All Tickers)"):
                        st.caption("This shows all options found before per-symbol deduplication")
                        full_cols = ["Ticker", "Type", "Strike", "DTE", "Spot", "OTM%", "Ask", "Model", "Edge%", "PoP%", "Delta", "IV", "Quality"]
                        st.dataframe(
                            df_results[full_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
                                "Edge%": st.column_config.NumberColumn("Edge%", format="%.1f%%"),
                                "PoP%": st.column_config.NumberColumn("PoP%", format="%.1f%%"),
                                "Quality": st.column_config.ProgressColumn("Quality", format="%.2f", min_value=0, max_value=1)
                            }
                        )
                    
                    # Best by ticker - now separated by class
                    st.markdown("---")
                    
                    col_single, col_etf = st.columns(2)
                    
                    with col_single:
                        st.subheader("📈 Top Single-Name Stocks")
                        single_names_top = best_per_symbol[~best_per_symbol["IsETF"]].head(15)
                        if not single_names_top.empty:
                            single_display = single_names_top[["Ticker", "Type", "Strike", "Edge%", "PoP%", "Quality"]].copy()
                            st.dataframe(
                                single_display,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                    "Edge%": st.column_config.NumberColumn("Edge%", format="%.1f%%"),
                                    "PoP%": st.column_config.NumberColumn("PoP%", format="%.1f%%"),
                                    "Quality": st.column_config.ProgressColumn("Quality", format="%.2f", min_value=0, max_value=1)
                                }
                            )
                        else:
                            st.info("No single-name stocks passed filters")
                    
                    with col_etf:
                        st.subheader("📊 Index ETFs")
                        etf_top = best_per_symbol[best_per_symbol["IsETF"]].head(5)
                        if not etf_top.empty:
                            etf_display = etf_top[["Ticker", "Type", "Strike", "Edge%", "PoP%", "Quality"]].copy()
                            st.dataframe(
                                etf_display,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                    "Edge%": st.column_config.NumberColumn("Edge%", format="%.1f%%"),
                                    "PoP%": st.column_config.NumberColumn("PoP%", format="%.1f%%"),
                                    "Quality": st.column_config.ProgressColumn("Quality", format="%.2f", min_value=0, max_value=1)
                                }
                            )
                        else:
                            if allow_index_etfs:
                                st.info("No index ETFs passed filters")
                            else:
                                st.info("Index ETFs excluded by preference")
                    
                    # Key
                    st.markdown("---")
                    st.markdown("""
                    ### 🔑 Option Hunter Key
                    
                    | Column | Meaning |
                    |--------|---------|
                    | **Edge%** | (Model Price - Ask) / Ask — higher = more underpriced |
                    | **PoP%** | Probability of Profit based on delta proxy |
                    | **Quality** | Composite score: 30% edge + 40% PoP + 30% liquidity |
                    | **OTM%** | How far out-of-the-money (negative = ITM) |
                    | **Model** | Black-Scholes theoretical fair value |
                    | **Class** | 📈 Stock (single name) or 📊 ETF (index) |
                    
                    **Color Coding:**
                    - 🟢 Green Edge% = Significant mispricing (>5%)
                    - 🔵 Blue PoP% = High probability (>50%)
                    - 🟡 Yellow Quality = Top-tier opportunity
                    
                    **Diversity Rules Applied:**
                    - Top 1 option per ticker (prevents single symbol domination)
                    - Single-name stocks only (ETFs have their own module: **📊 ETF Opts**)
                    """)
                    
                else:
                    progress_bar.empty()
                    st.warning("No underpriced options found matching your criteria. Try adjusting filters:")
                    st.markdown("""
                    - Lower the Min Edge %
                    - Increase Max Spread %
                    - Lower the Min PoP %
                    - Try a different DTE range
                    """)
        else:
            st.info("🔍 **Configure your scan parameters above**, then click **Scan** to find underpriced options across 60+ liquid tickers.")
            
            st.markdown("""
            ### How Option Hunter Works
            
            1. **Scans** 70+ highly liquid single-name stocks for options
            2. **Filters** by liquidity (spread %, volume, open interest)  
            3. **Calculates** theoretical value using Black-Scholes model
            4. **Identifies** options trading below fair value (positive edge)
            5. **Ranks** by Quality Score (edge + PoP + liquidity)
            
            **Best For:**
            - Finding cheap convexity on individual stocks
            - Directional plays with edge
            - Identifying mispriced volatility
            
            💡 *For ETF options, use the **📊 ETF Opts** module*
            """)

    # ==========================================
    # MODULE F: ET-EFFER HUNTER (ETF Scanner)
    # ==========================================
    elif mode == "📊 ETF Hunt":
        st.title("📊 ET-effer Hunter: ETF Scanner")
        st.caption("Scan top ETFs using the Ryan Model — momentum, squeeze, trap detection")
        
        with st.expander("🛠️ Scanner Settings", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                etf_scan_window = st.slider("Lookback Window", 10, 50, 20, key="etf_scan_window")
            with sc2:
                etf_sqz_thresh = st.slider("Squeeze Threshold", 0.10, 0.40, 0.25, key="etf_sqz_thresh")
            with sc3:
                etf_min_confidence = st.slider("Min Confidence %", 50, 95, 80, key="etf_min_conf")
        
        # Top 100 ETFs by Volume/AUM
        ETF_TICKER_LIST = [
            # Broad Market ETFs
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "VEA", "EFA", "VWO",
            "IEMG", "VTV", "VUG", "IJH", "IJR", "VIG", "SCHD", "VYM", "ITOT", "IXUS",
            # Sector ETFs
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
            "XLC", "VGT", "VFH", "VHT", "VCR", "VDC", "VIS", "VAW", "VNQ", "VPU",
            # Thematic / Industry
            "ARKK", "ARKG", "ARKW", "ARKF", "SOXX", "SMH", "XBI", "IBB", "XHB", "XRT",
            "KRE", "XOP", "OIH", "GDX", "GDXJ", "SLV", "GLD", "USO", "UNG", "JETS",
            # Fixed Income
            "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "BND", "AGG", "TIP", "MUB",
            # Leveraged / Inverse (High Vol)
            "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SOXL", "SOXS", "LABU", "LABD", "FAS",
            "FAZ", "ERX", "ERY", "NUGT", "DUST", "UVXY", "SVXY", "TNA", "TZA", "UDOW",
            # International
            "EWJ", "EWZ", "EWG", "EWU", "FXI", "MCHI", "INDA", "EWT", "EWY", "EWA",
            # Commodity / Currency
            "DBC", "PDBC", "GSG", "DBA", "UUP", "FXE", "FXY", "FXB",
        ]
        
        @st.cache_data(ttl=600)
        def etf_batch_process(tickers, scan_window, scan_sqz_thresh):
            results = []
            data_batch = yf.download(tickers + ["SPY"], period="6mo", progress=False)
            
            def get_series(df, symbol, col):
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        return df[col][symbol]
                    else:
                        return df[col]
                except Exception:
                    return None
            
            for symbol in tickers:
                try:
                    closes = get_series(data_batch, symbol, "Close")
                    highs = get_series(data_batch, symbol, "High")
                    lows = get_series(data_batch, symbol, "Low")
                    spy_closes = get_series(data_batch, "SPY", "Close")
                    
                    if closes is None or len(closes.dropna()) < 60:
                        continue
                    
                    closes = closes.dropna()
                    highs = highs.dropna() if highs is not None else closes
                    lows = lows.dropna() if lows is not None else closes
                    spy_closes = spy_closes.dropna() if spy_closes is not None else closes
                    
                    # Ryan Model metrics
                    sma = closes.rolling(window=scan_window).mean()
                    std = closes.rolling(window=scan_window).std()
                    upper = sma + (std * 2.0)
                    lower = sma - (std * 2.0)
                    bandwidth = (upper - lower) / sma
                    
                    current_price = closes.iloc[-1]
                    current_bw = bandwidth.iloc[-1]
                    is_squeeze = current_bw < scan_sqz_thresh
                    
                    squeeze_depth = 0
                    if is_squeeze:
                        squeeze_depth = (scan_sqz_thresh - current_bw) / scan_sqz_thresh
                        squeeze_depth = min(max(squeeze_depth, 0), 1)
                    
                    trend_bias = "BULLISH" if current_price > sma.iloc[-1] else "BEARISH"
                    
                    ema_9 = closes.ewm(span=9, adjust=False).mean()
                    ema_21 = closes.ewm(span=21, adjust=False).mean()
                    ema_signal = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"
                    
                    stock_pct = closes.pct_change(60).iloc[-1]
                    spy_pct = spy_closes.pct_change(60).iloc[-1] if len(spy_closes) >= 60 else 0
                    rs_ratio = stock_pct - spy_pct
                    
                    # Trap detection
                    curr_high = highs.iloc[-1]
                    curr_low = lows.iloc[-1]
                    day_range = curr_high - curr_low
                    
                    if day_range == 0:
                        range_position = 0.5
                    else:
                        range_position = (current_price - curr_low) / day_range
                    
                    if range_position > 0.8:
                        intraday_health = "POWER"
                    elif range_position < 0.2:
                        intraday_health = "WEAK"
                    elif range_position < 0.5 and trend_bias == "BULLISH":
                        intraday_health = "FADING"
                    else:
                        intraday_health = "NEUTRAL"
                    
                    # Confidence calculation
                    confidence = 0
                    if trend_bias == "BULLISH" and ema_signal == "BULLISH":
                        confidence += 35
                    elif trend_bias == "BEARISH" and ema_signal == "BEARISH":
                        confidence += 35
                    else:
                        confidence += 15
                    
                    if is_squeeze:
                        confidence += 15 + (squeeze_depth * 10)
                    
                    if trend_bias == "BULLISH" and rs_ratio > 0:
                        confidence += 20
                    elif trend_bias == "BEARISH" and rs_ratio < 0:
                        confidence += 20
                    
                    if intraday_health in ["FADING", "WEAK"]:
                        confidence -= 15
                    
                    confidence = min(max(int(confidence), 0), 99)
                    
                    # Action signal
                    action_signal = "WAIT"
                    if confidence >= 80 and intraday_health == "POWER" and trend_bias == "BULLISH":
                        action_signal = "✅ BUY"
                    elif confidence >= 80 and intraday_health == "WEAK" and trend_bias == "BEARISH":
                        action_signal = "🔻 SHORT"
                    
                    results.append({
                        "Ticker": symbol,
                        "Price": current_price,
                        "Action": action_signal,
                        "Trend": trend_bias,
                        "Momentum": ema_signal,
                        "Squeeze": "COILED" if is_squeeze else "LOOSE",
                        "Bandwidth": current_bw,
                        "Confidence": confidence,
                        "RS_vs_SPY": rs_ratio,
                        "Health": intraday_health,
                    })
                except Exception:
                    continue
            
            return pd.DataFrame(results)
        
        if st.button("🚀 Scan ETF Universe", key="etf_hunter_btn"):
            progress_bar = st.progress(0, text="Initializing ETF scan...")
            
            with st.spinner(f"Scanning {len(ETF_TICKER_LIST)} ETFs..."):
                progress_bar.progress(30, text="Downloading ETF data...")
                df_etf_results = etf_batch_process(ETF_TICKER_LIST, etf_scan_window, etf_sqz_thresh)
                progress_bar.progress(90, text="Applying Ryan Model...")
                
                if df_etf_results is not None and not df_etf_results.empty:
                    progress_bar.progress(100, text="Scan Complete.")
                    progress_bar.empty()
                    
                    etf_longs = df_etf_results[
                        (df_etf_results["Trend"] == "BULLISH") &
                        (df_etf_results["Momentum"] == "BULLISH") &
                        (df_etf_results["Squeeze"] == "COILED") &
                        (df_etf_results["Confidence"] >= etf_min_confidence)
                    ]
                    
                    etf_shorts = df_etf_results[
                        (df_etf_results["Trend"] == "BEARISH") &
                        (df_etf_results["Momentum"] == "BEARISH") &
                        (df_etf_results["Squeeze"] == "COILED") &
                        (df_etf_results["Confidence"] >= etf_min_confidence)
                    ]
                    
                    # Long ETFs
                    st.subheader(f"🟢 Long ETF Setups ({len(etf_longs)})")
                    if not etf_longs.empty:
                        st.dataframe(
                            etf_longs,
                            use_container_width=True,
                            column_config={
                                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                "Bandwidth": st.column_config.NumberColumn("Bandwidth", format="%.4f"),
                                "RS_vs_SPY": st.column_config.NumberColumn("RS vs SPY", format="%.2f%%"),
                                "Confidence": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100)
                            },
                            column_order=("Ticker", "Action", "Confidence", "Health", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                            hide_index=True
                        )
                    else:
                        st.info(f"No Long ETF setups found (Conf > {etf_min_confidence}% + Coiled)")
                    
                    st.markdown("---")
                    
                    # Short ETFs
                    st.subheader(f"🔴 Short ETF Setups ({len(etf_shorts)})")
                    if not etf_shorts.empty:
                        st.dataframe(
                            etf_shorts,
                            use_container_width=True,
                            column_config={
                                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                "Bandwidth": st.column_config.NumberColumn("Bandwidth", format="%.4f"),
                                "RS_vs_SPY": st.column_config.NumberColumn("RS vs SPY", format="%.2f%%"),
                                "Confidence": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100)
                            },
                            column_order=("Ticker", "Action", "Confidence", "Health", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                            hide_index=True
                        )
                    else:
                        st.info(f"No Short ETF setups found (Conf > {etf_min_confidence}% + Coiled)")
                    
                    # Full results
                    with st.expander("📂 View Full ETF Scan Results"):
                        st.dataframe(
                            df_etf_results.sort_values("Confidence", ascending=False),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                "Confidence": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100)
                            }
                        )
                    
                    st.markdown("---")
                    st.markdown("""
                    ### 🔑 ETF Hunter Key
                    
                    | Signal | Meaning |
                    |--------|---------|
                    | ✅ BUY | Conf ≥80% + POWER health + Bullish |
                    | 🔻 SHORT | Conf ≥80% + WEAK health + Bearish |
                    | WAIT | Setup forming but not actionable yet |
                    
                    **ETF Categories Scanned:** Broad Market, Sectors, Thematic, Fixed Income, Leveraged, International
                    """)
                else:
                    progress_bar.empty()
                    st.error("ETF scan returned no data. Check API connection.")
        else:
            st.info("📊 **Ready to scan 100+ ETFs** using the Ryan Model. Click to find momentum setups across sectors, themes, and asset classes.")

    # ==========================================
    # MODULE G: ET-EFFER OPTIONS (ETF Options Scanner)
    # ==========================================
    elif mode == "📊 ETF Opts":
        st.title("📊 ET-effer Options: ETF Options Scanner")
        st.caption("Find underpriced options on ETFs — sectors, themes, indices")
        
        st.markdown("### ⚙️ Scanner Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            etf_dte_range = st.selectbox("Days to Expiration", ["7-14 days", "14-30 days", "30-60 days", "60-90 days"], index=1, key="etf_dte")
        with col2:
            etf_direction = st.selectbox("Direction", ["Bullish (Calls)", "Bearish (Puts)", "Both"], key="etf_dir")
        with col3:
            etf_min_pop = st.slider("Min PoP %", 30, 80, 50, key="etf_pop")
        with col4:
            etf_max_spread = st.slider("Max Spread %", 5, 30, 15, key="etf_spread")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            etf_min_edge = st.slider("Min Edge %", 0, 30, 5, key="etf_edge")
        with col6:
            etf_max_otm = st.slider("Max OTM %", 5, 30, 15, key="etf_otm")
        with col7:
            etf_rf = st.number_input("Risk Free %", value=4.5, step=0.1, key="etf_rf")
        with col8:
            etf_max_results = st.selectbox("Top Results", [5, 10, 20, 50], index=1, key="etf_max")
        
        # Parse DTE
        etf_dte_map = {
            "7-14 days": (7, 14),
            "14-30 days": (14, 30),
            "30-60 days": (30, 60),
            "60-90 days": (60, 90)
        }
        etf_min_dte, etf_max_dte = etf_dte_map[etf_dte_range]
        
        # ETF Options Universe (most liquid ETFs for options)
        ETF_OPTIONS_LIST = [
            # Index ETFs (highest options liquidity)
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EFA", "EEM",
            # Sector ETFs
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
            # Industry ETFs
            "SOXX", "SMH", "XBI", "IBB", "XHB", "XRT", "KRE", "XOP", "OIH", "GDX",
            # Commodity ETFs
            "GLD", "SLV", "USO", "UNG",
            # Fixed Income
            "TLT", "IEF", "HYG", "LQD", "JNK",
            # Leveraged (High Vol = great for options)
            "TQQQ", "SQQQ", "SOXL", "SOXS", "UVXY", "TNA", "TZA",
            # Thematic
            "ARKK", "ARKG", "JETS",
            # International
            "EWZ", "FXI", "EWJ", "MCHI",
        ]
        
        @st.cache_data(ttl=300, show_spinner=False)
        def scan_etf_options(tickers, min_dte, max_dte, direction, min_pop_thresh, max_spread, min_edge_pct, max_otm, rf):
            """Scan ETFs for underpriced options"""
            results = []
            hydra_scan = HydraEngine()
            
            for ticker in tickers:
                try:
                    spot, _ = hydra_scan.get_spot(ticker)
                    if spot is None or spot <= 0:
                        continue
                    
                    exps = hydra_scan.get_expirations(ticker)
                    if not exps:
                        continue
                    
                    target_exp = None
                    target_dte = None
                    for exp in exps:
                        try:
                            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                            dte = (exp_date - date.today()).days
                            if min_dte <= dte <= max_dte:
                                target_exp = exp
                                target_dte = dte
                                break
                        except:
                            continue
                    
                    if not target_exp:
                        continue
                    
                    chain, _ = hydra_scan.get_chain(ticker, target_exp, spot)
                    if chain is None or chain.empty:
                        continue
                    
                    chain["spread"] = chain["ask"] - chain["bid"]
                    chain["spread_pct"] = np.where(chain["bid"] > 0, (chain["spread"] / chain["bid"]) * 100, 100)
                    
                    if direction == "Bullish (Calls)":
                        chain = chain[chain["type"] == "call"]
                    elif direction == "Bearish (Puts)":
                        chain = chain[chain["type"] == "put"]
                    
                    liquid_chain = chain[
                        (chain["spread_pct"] <= max_spread) &
                        (chain["bid"] > 0.05) &
                        ((chain["volume"] >= 10) | (chain["openInterest"] >= 100))
                    ].copy()
                    
                    if liquid_chain.empty:
                        continue
                    
                    liquid_chain["otm_pct"] = np.where(
                        liquid_chain["type"] == "call",
                        ((liquid_chain["strike"] - spot) / spot) * 100,
                        ((spot - liquid_chain["strike"]) / spot) * 100
                    )
                    
                    liquid_chain = liquid_chain[(liquid_chain["otm_pct"] >= -5) & (liquid_chain["otm_pct"] <= max_otm)]
                    
                    if liquid_chain.empty:
                        continue
                    
                    T = target_dte / 365.0
                    
                    for _, row in liquid_chain.iterrows():
                        try:
                            strike = row["strike"]
                            opt_type = row["type"]
                            iv = row.get("impliedVolatility", 0.3)
                            if iv <= 0 or iv > 3:
                                iv = 0.3
                            
                            kernel = PricingKernel(spot, strike, T, rf / 100, iv, opt_type)
                            model_price = kernel.price_bsm()
                            
                            ask_price = row["ask"]
                            edge = model_price - ask_price
                            edge_pct = (edge / ask_price * 100) if ask_price > 0 else 0
                            
                            if edge_pct < min_edge_pct:
                                continue
                            
                            greeks = kernel.get_greeks()
                            delta = greeks.delta
                            
                            pop_raw = abs(delta)
                            pop_adjusted = pop_raw * (1 - row["otm_pct"] / 100 * 0.5)
                            pop_adjusted = max(0.05, min(0.95, pop_adjusted))
                            
                            if pop_adjusted * 100 < min_pop_thresh:
                                continue
                            
                            quality = (edge_pct / 100) * 0.3 + pop_adjusted * 0.4 + (1 - row["spread_pct"] / max_spread) * 0.3
                            
                            results.append({
                                "Ticker": ticker,
                                "Type": opt_type.upper(),
                                "Strike": strike,
                                "DTE": target_dte,
                                "Spot": spot,
                                "OTM%": row["otm_pct"],
                                "Ask": row["ask"],
                                "Model": model_price,
                                "Edge%": edge_pct,
                                "PoP%": pop_adjusted * 100,
                                "Delta": delta,
                                "IV": iv * 100,
                                "Spread%": row["spread_pct"],
                                "Quality": quality
                            })
                        except:
                            continue
                except:
                    continue
            
            return pd.DataFrame(results)
        
        if st.button("🚀 Scan ETF Options", type="primary", use_container_width=True, key="etf_opt_btn"):
            progress_bar = st.progress(0, text="Initializing ETF options scan...")
            
            with st.spinner(f"Scanning {len(ETF_OPTIONS_LIST)} ETFs for mispriced options..."):
                progress_bar.progress(20, text="Fetching options chains...")
                
                df_etf_opts = scan_etf_options(
                    ETF_OPTIONS_LIST, etf_min_dte, etf_max_dte, etf_direction,
                    etf_min_pop, etf_max_spread, etf_min_edge, etf_max_otm, etf_rf
                )
                
                progress_bar.progress(90, text="Ranking opportunities...")
                
                if df_etf_opts is not None and not df_etf_opts.empty:
                    progress_bar.progress(100, text="Scan complete!")
                    progress_bar.empty()
                    
                    # Per-symbol best + global ranking
                    df_etf_opts = df_etf_opts.sort_values("Quality", ascending=False)
                    best_per_etf = df_etf_opts.groupby("Ticker").head(1)
                    final_etf_results = best_per_etf.sort_values("Quality", ascending=False).head(etf_max_results)
                    
                    st.markdown("---")
                    st.markdown("### 📊 ETF Options Scan Results")
                    
                    stat1, stat2, stat3, stat4 = st.columns(4)
                    with stat1:
                        st.metric("Top ETF Ideas", len(final_etf_results))
                    with stat2:
                        st.metric("Total Options Found", len(df_etf_opts))
                    with stat3:
                        avg_edge = final_etf_results["Edge%"].mean() if len(final_etf_results) > 0 else 0
                        st.metric("Avg Edge %", f"{avg_edge:.1f}%")
                    with stat4:
                        avg_pop = final_etf_results["PoP%"].mean() if len(final_etf_results) > 0 else 0
                        st.metric("Avg PoP %", f"{avg_pop:.1f}%")
                    
                    st.markdown("---")
                    st.subheader(f"🎯 Top Underpriced ETF Options ({len(final_etf_results)} ETFs)")
                    
                    display_cols = ["Ticker", "Type", "Strike", "DTE", "Spot", "OTM%", "Ask", "Model", "Edge%", "PoP%", "Delta", "IV", "Quality"]
                    
                    st.dataframe(
                        final_etf_results[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                            "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
                            "OTM%": st.column_config.NumberColumn("OTM%", format="%.1f%%"),
                            "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                            "Model": st.column_config.NumberColumn("Model", format="$%.2f"),
                            "Edge%": st.column_config.NumberColumn("Edge%", format="%.1f%%"),
                            "PoP%": st.column_config.NumberColumn("PoP%", format="%.1f%%"),
                            "Delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                            "IV": st.column_config.NumberColumn("IV", format="%.1f%%"),
                            "Quality": st.column_config.ProgressColumn("Quality", format="%.2f", min_value=0, max_value=1)
                        }
                    )
                    
                    with st.expander("📋 Full ETF Options Results"):
                        st.dataframe(
                            df_etf_opts[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
                                "Edge%": st.column_config.NumberColumn("Edge%", format="%.1f%%"),
                                "PoP%": st.column_config.NumberColumn("PoP%", format="%.1f%%"),
                                "Quality": st.column_config.ProgressColumn("Quality", format="%.2f", min_value=0, max_value=1)
                            }
                        )
                    
                    st.markdown("---")
                    st.markdown("""
                    ### 🔑 ETF Options Key
                    
                    | Column | Meaning |
                    |--------|---------|
                    | **Edge%** | (Model - Ask) / Ask — higher = more underpriced |
                    | **PoP%** | Probability of Profit via delta proxy |
                    | **Quality** | 30% edge + 40% PoP + 30% liquidity |
                    
                    **ETF Categories:** Index, Sector, Industry, Commodity, Fixed Income, Leveraged, Thematic
                    """)
                else:
                    progress_bar.empty()
                    st.warning("No underpriced ETF options found. Try adjusting filters.")
        else:
            st.info("📊 **Scan 50+ liquid ETFs** for underpriced options across indices, sectors, commodities, and leveraged products.")
            
            st.markdown("""
            ### ETF Options Scanner
            
            Scans the most liquid ETFs for mispriced options:
            - **Index ETFs:** SPY, QQQ, IWM, DIA
            - **Sector ETFs:** XLF, XLK, XLE, XLV, etc.
            - **Commodity:** GLD, SLV, USO
            - **Fixed Income:** TLT, HYG, LQD
            - **Leveraged:** TQQQ, SOXL, UVXY (high vol = cheap convexity)
            
            💡 *For single-name stock options, use the **🔍 Opt Hunt** module*
            """)

    # ==========================================
    # MODULE H: SHORTIES (Short Squeeze Scanner)
    # ==========================================
    elif mode == "🩳 Shorties":
        st.title("🩳 Shorties: Short Squeeze Scanner")
        st.caption("Find the next AMC/GME — high short interest stocks ready to squeeze")
        
        st.markdown("### ⚙️ Squeeze Scanner Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_short_pct = st.slider("Min Short Interest %", 5, 30, 10, help="Minimum short % of float")
        with col2:
            min_rel_volume = st.slider("Min Relative Volume", 1.0, 5.0, 1.5, help="Today's volume vs 20-day avg")
        with col3:
            min_squeeze_score = st.slider("Min Squeeze Score", 30, 80, 50)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_price = st.number_input("Max Stock Price ($)", value=100.0, step=10.0, help="Filter out expensive stocks")
        with col5:
            min_price = st.number_input("Min Stock Price ($)", value=1.0, step=0.5, help="Filter penny stocks")
        with col6:
            lookback_days = st.selectbox("Momentum Period", [5, 10, 20, 30], index=1)
        
        # Short Squeeze Universe - Known high short interest + meme + small caps
        SQUEEZE_UNIVERSE = [
            # Classic Meme / High Short Interest
            "GME", "AMC", "BBBY", "KOSS", "EXPR", "BB", "NOK", "NAKD", "SNDL", "TLRY",
            "CLOV", "WISH", "WKHS", "GOEV", "RIDE", "NKLA", "HYMC", "MULN", "FFIE", "VINE",
            # Biotech Squeezes (often heavily shorted)
            "ATOS", "PROG", "BBIG", "SPRT", "IRNT", "SDC", "OPAD", "TMC", "VIR", "BKKT",
            # Recent IPO / SPAC that get shorted
            "RIVN", "LCID", "PLTR", "SOFI", "HOOD", "COIN", "AFRM", "UPST", "RBLX", "U",
            # EV / Clean Energy (high short interest sector)
            "TSLA", "NIO", "XPEV", "LI", "FSR", "ARVL", "EVGO", "BLNK", "CHPT", "QS",
            # Crypto Exposed
            "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BTBT", "BITF", "HIVE", "CIFR", "CORZ",
            # Retail / Consumer (often shorted)
            "BBWI", "GPS", "URBN", "ANF", "EXPR", "PRTY", "BIG", "BGFV", "DDS", "M",
            # Tech Small/Mid (volatile, often shorted)
            "SNAP", "PINS", "ETSY", "ROKU", "PTON", "ZM", "DOCU", "FVRR", "UPWK", "OPEN",
            # Pharma / Healthcare (binary events)
            "NVAX", "MRNA", "BNTX", "SAVA", "RETA", "VERU", "OCGN", "INO", "SRNE", "VXRT",
            # Real Estate / Mortgage (rate sensitive, shorted)
            "RKT", "UWMC", "CLOV", "LMND", "ROOT", "OPEN", "RDFN", "CVNA", "VRM", "SFT",
            # Miscellaneous High Vol
            "AI", "IONQ", "DNA", "JOBY", "ARQQ", "QBTS", "RGTI", "IonQ", "PATH", "DOCN",
            # Additional Squeeze Candidates
            "SPCE", "DKNG", "PENN", "CHWY", "W", "BYND", "CRSR", "PRTS", "SKLZ", "BODY",
        ]
        
        # Remove duplicates
        SQUEEZE_UNIVERSE = list(set(SQUEEZE_UNIVERSE))
        
        @st.cache_data(ttl=300, show_spinner=False)
        def scan_short_squeeze_candidates(tickers, min_short, min_rel_vol, min_price_filter, max_price_filter, momentum_days):
            """Scan for short squeeze candidates using multiple signals"""
            results = []
            
            # Batch download price data
            try:
                price_data = yf.download(tickers, period="3mo", progress=False)
            except:
                return pd.DataFrame()
            
            for ticker in tickers:
                try:
                    # Get individual ticker info for short data
                    tk = yf.Ticker(ticker)
                    info = tk.info
                    
                    # Extract short interest metrics (may not always be available)
                    short_pct = info.get("shortPercentOfFloat", 0) or 0
                    if short_pct > 1:  # Sometimes returned as percentage, sometimes as decimal
                        short_pct = short_pct
                    else:
                        short_pct = short_pct * 100
                    
                    short_ratio = info.get("shortRatio", 0) or 0  # Days to cover
                    float_shares = info.get("floatShares", 0) or 0
                    shares_short = info.get("sharesShort", 0) or 0
                    
                    # Get price data
                    if isinstance(price_data.columns, pd.MultiIndex):
                        closes = price_data["Close"][ticker].dropna()
                        volumes = price_data["Volume"][ticker].dropna()
                        highs = price_data["High"][ticker].dropna()
                        lows = price_data["Low"][ticker].dropna()
                    else:
                        closes = price_data["Close"].dropna()
                        volumes = price_data["Volume"].dropna()
                        highs = price_data["High"].dropna()
                        lows = price_data["Low"].dropna()
                    
                    if len(closes) < 30:
                        continue
                    
                    current_price = closes.iloc[-1]
                    
                    # Price filter
                    if current_price < min_price_filter or current_price > max_price_filter:
                        continue
                    
                    # Calculate key metrics
                    avg_volume_20 = volumes.tail(20).mean()
                    current_volume = volumes.iloc[-1]
                    rel_volume = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
                    
                    # Momentum (% change over lookback period)
                    if len(closes) >= momentum_days:
                        momentum_pct = ((current_price / closes.iloc[-momentum_days]) - 1) * 100
                    else:
                        momentum_pct = 0
                    
                    # Volatility (20-day realized vol annualized)
                    returns = closes.pct_change().dropna()
                    volatility = returns.tail(20).std() * np.sqrt(252) * 100 if len(returns) >= 20 else 0
                    
                    # Beta estimation (vs simple momentum)
                    beta = info.get("beta", 1.0) or 1.0
                    
                    # 52-week metrics
                    high_52w = highs.tail(252).max() if len(highs) >= 252 else highs.max()
                    low_52w = lows.tail(252).min() if len(lows) >= 252 else lows.min()
                    pct_from_high = ((current_price / high_52w) - 1) * 100 if high_52w > 0 else 0
                    pct_from_low = ((current_price / low_52w) - 1) * 100 if low_52w > 0 else 0
                    
                    # Squeeze detection signals
                    # 1. Bollinger Band Squeeze
                    sma_20 = closes.rolling(20).mean()
                    std_20 = closes.rolling(20).std()
                    bandwidth = (std_20.iloc[-1] * 2 / sma_20.iloc[-1]) if sma_20.iloc[-1] > 0 else 0
                    is_squeezed = bandwidth < 0.15  # Tight bands
                    
                    # 2. Price breaking out
                    sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else sma_20.iloc[-1]
                    is_breaking_out = current_price > sma_50 and momentum_pct > 0
                    
                    # 3. Volume surge
                    volume_surge = rel_volume > 2.0
                    
                    # 4. Recent higher lows (accumulation)
                    recent_lows = lows.tail(10)
                    higher_lows = all(recent_lows.iloc[i] <= recent_lows.iloc[i+1] for i in range(min(3, len(recent_lows)-1)))
                    
                    # ======== SQUEEZE SCORE CALCULATION ========
                    # Weighted scoring system
                    squeeze_score = 0
                    
                    # Short Interest Component (0-30 points)
                    if short_pct >= 30:
                        squeeze_score += 30
                    elif short_pct >= 20:
                        squeeze_score += 25
                    elif short_pct >= 15:
                        squeeze_score += 20
                    elif short_pct >= 10:
                        squeeze_score += 15
                    elif short_pct >= 5:
                        squeeze_score += 10
                    
                    # Days to Cover Component (0-15 points)
                    if short_ratio >= 5:
                        squeeze_score += 15
                    elif short_ratio >= 3:
                        squeeze_score += 10
                    elif short_ratio >= 2:
                        squeeze_score += 5
                    
                    # Relative Volume Component (0-20 points)
                    if rel_volume >= 3:
                        squeeze_score += 20
                    elif rel_volume >= 2:
                        squeeze_score += 15
                    elif rel_volume >= 1.5:
                        squeeze_score += 10
                    elif rel_volume >= 1.2:
                        squeeze_score += 5
                    
                    # Momentum Component (0-15 points)
                    if momentum_pct >= 20:
                        squeeze_score += 15
                    elif momentum_pct >= 10:
                        squeeze_score += 12
                    elif momentum_pct >= 5:
                        squeeze_score += 8
                    elif momentum_pct >= 0:
                        squeeze_score += 4
                    
                    # Volatility Component (0-10 points) - Higher vol = more explosive
                    if volatility >= 100:
                        squeeze_score += 10
                    elif volatility >= 75:
                        squeeze_score += 8
                    elif volatility >= 50:
                        squeeze_score += 5
                    
                    # Technical Signals (0-10 points)
                    if is_squeezed:
                        squeeze_score += 3
                    if is_breaking_out:
                        squeeze_score += 4
                    if volume_surge:
                        squeeze_score += 3
                    
                    # Determine direction bias
                    if momentum_pct > 5 and rel_volume > 1.5 and is_breaking_out:
                        direction = "🚀 MOON"
                        action = "BUY SQUEEZE"
                    elif momentum_pct < -5 and rel_volume > 1.5:
                        direction = "💥 CRATER"
                        action = "SHORT/PUTS"
                    elif squeeze_score >= 60:
                        direction = "⚡ LOADING"
                        action = "WATCH"
                    else:
                        direction = "😴 DORMANT"
                        action = "WAIT"
                    
                    # Float category
                    if float_shares > 0:
                        if float_shares < 10_000_000:
                            float_cat = "🔥 MICRO"
                        elif float_shares < 50_000_000:
                            float_cat = "⚡ SMALL"
                        elif float_shares < 200_000_000:
                            float_cat = "📊 MED"
                        else:
                            float_cat = "🏢 LARGE"
                    else:
                        float_cat = "❓ UNK"
                    
                    results.append({
                        "Ticker": ticker,
                        "Price": current_price,
                        "Action": action,
                        "Direction": direction,
                        "Short%": short_pct,
                        "DTC": short_ratio,  # Days to Cover
                        "RelVol": rel_volume,
                        "Mom%": momentum_pct,
                        "Vol%": volatility,
                        "Float": float_cat,
                        "Squeeze": squeeze_score,
                        "From52H%": pct_from_high,
                        "From52L%": pct_from_low,
                        "Beta": beta
                    })
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(results)
        
        # Run Scanner
        if st.button("🩳 Find Short Squeeze Candidates", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing squeeze scanner...")
            
            with st.spinner(f"Scanning {len(SQUEEZE_UNIVERSE)} potential squeeze candidates..."):
                progress_bar.progress(20, text="Fetching short interest data...")
                
                df_squeeze = scan_short_squeeze_candidates(
                    SQUEEZE_UNIVERSE,
                    min_short_pct,
                    min_rel_volume,
                    min_price,
                    max_price,
                    lookback_days
                )
                
                progress_bar.progress(80, text="Calculating squeeze scores...")
                
                if df_squeeze is not None and not df_squeeze.empty:
                    # Filter by minimum squeeze score
                    df_squeeze = df_squeeze[df_squeeze["Squeeze"] >= min_squeeze_score]
                    df_squeeze = df_squeeze.sort_values("Squeeze", ascending=False)
                    
                    progress_bar.progress(100, text="Scan complete!")
                    progress_bar.empty()
                    
                    if len(df_squeeze) > 0:
                        # Stats
                        st.markdown("---")
                        st.markdown("### 📊 Squeeze Scanner Results")
                        
                        stat1, stat2, stat3, stat4 = st.columns(4)
                        with stat1:
                            st.metric("Candidates Found", len(df_squeeze))
                        with stat2:
                            moon_count = len(df_squeeze[df_squeeze["Direction"] == "🚀 MOON"])
                            st.metric("🚀 Moon Signals", moon_count)
                        with stat3:
                            avg_short = df_squeeze["Short%"].mean()
                            st.metric("Avg Short %", f"{avg_short:.1f}%")
                        with stat4:
                            avg_squeeze = df_squeeze["Squeeze"].mean()
                            st.metric("Avg Squeeze Score", f"{avg_squeeze:.0f}")
                        
                        st.markdown("---")
                        
                        # 🚀 MOON CANDIDATES
                        moon_df = df_squeeze[df_squeeze["Direction"] == "🚀 MOON"]
                        st.subheader(f"🚀 MOON CANDIDATES ({len(moon_df)})")
                        st.caption("Active squeeze in progress — high momentum + volume surge")
                        
                        if not moon_df.empty:
                            st.dataframe(
                                moon_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                    "Short%": st.column_config.NumberColumn("Short%", format="%.1f%%"),
                                    "DTC": st.column_config.NumberColumn("Days Cover", format="%.1f"),
                                    "RelVol": st.column_config.NumberColumn("Rel Vol", format="%.1fx"),
                                    "Mom%": st.column_config.NumberColumn("Mom%", format="%.1f%%"),
                                    "Vol%": st.column_config.NumberColumn("Vol%", format="%.0f%%"),
                                    "Squeeze": st.column_config.ProgressColumn("Squeeze", format="%d", min_value=0, max_value=100),
                                    "From52H%": st.column_config.NumberColumn("vs 52H", format="%.1f%%"),
                                    "From52L%": st.column_config.NumberColumn("vs 52L", format="%.1f%%"),
                                    "Beta": st.column_config.NumberColumn("Beta", format="%.2f")
                                },
                                column_order=["Ticker", "Action", "Direction", "Price", "Squeeze", "Short%", "DTC", "RelVol", "Mom%", "Vol%", "Float"]
                            )
                        else:
                            st.info("No active moon signals right now. Check the Loading/Watch list below.")
                        
                        st.markdown("---")
                        
                        # ⚡ LOADING / WATCH
                        watch_df = df_squeeze[df_squeeze["Direction"].isin(["⚡ LOADING", "😴 DORMANT"])]
                        st.subheader(f"⚡ LOADING — Watch List ({len(watch_df)})")
                        st.caption("Building pressure — could squeeze soon")
                        
                        if not watch_df.empty:
                            st.dataframe(
                                watch_df.head(20),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                    "Short%": st.column_config.NumberColumn("Short%", format="%.1f%%"),
                                    "DTC": st.column_config.NumberColumn("Days Cover", format="%.1f"),
                                    "RelVol": st.column_config.NumberColumn("Rel Vol", format="%.1fx"),
                                    "Mom%": st.column_config.NumberColumn("Mom%", format="%.1f%%"),
                                    "Squeeze": st.column_config.ProgressColumn("Squeeze", format="%d", min_value=0, max_value=100)
                                },
                                column_order=["Ticker", "Action", "Direction", "Price", "Squeeze", "Short%", "DTC", "RelVol", "Mom%", "Float"]
                            )
                        
                        st.markdown("---")
                        
                        # 💥 CRATER CANDIDATES (Potential shorts)
                        crater_df = df_squeeze[df_squeeze["Direction"] == "💥 CRATER"]
                        if not crater_df.empty:
                            st.subheader(f"💥 CRATER — Short/Put Candidates ({len(crater_df)})")
                            st.caption("Squeeze failed or reversing — potential downside")
                            st.dataframe(
                                crater_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                    "Short%": st.column_config.NumberColumn("Short%", format="%.1f%%"),
                                    "Mom%": st.column_config.NumberColumn("Mom%", format="%.1f%%"),
                                    "Squeeze": st.column_config.ProgressColumn("Squeeze", format="%d", min_value=0, max_value=100)
                                },
                                column_order=["Ticker", "Action", "Direction", "Price", "Squeeze", "Short%", "Mom%", "RelVol", "Float"]
                            )
                        
                        # Full Results
                        with st.expander("📋 Full Squeeze Scanner Results"):
                            st.dataframe(
                                df_squeeze,
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        # Key
                        st.markdown("---")
                        st.markdown("""
                        ### 🔑 Shorties Squeeze Key
                        
                        | Signal | Meaning |
                        |--------|---------|
                        | 🚀 **MOON** | Active squeeze — momentum + volume surge + breakout |
                        | ⚡ **LOADING** | High squeeze score, waiting for catalyst |
                        | 💥 **CRATER** | Reversal/failure — potential short opportunity |
                        | 😴 **DORMANT** | High short interest but no momentum yet |
                        
                        | Metric | What It Means |
                        |--------|---------------|
                        | **Short%** | Percent of float sold short (>20% = high) |
                        | **DTC** | Days to Cover — how long to close shorts (>3 = trapped) |
                        | **RelVol** | Today's volume vs 20-day avg (>2x = surge) |
                        | **Mom%** | Price momentum over selected period |
                        | **Vol%** | Annualized volatility (higher = more explosive) |
                        | **Float** | 🔥MICRO (<10M) ⚡SMALL (<50M) 📊MED (<200M) 🏢LARGE |
                        
                        **Squeeze Score Components:**
                        - Short Interest: 0-30 pts
                        - Days to Cover: 0-15 pts
                        - Relative Volume: 0-20 pts
                        - Momentum: 0-15 pts
                        - Volatility: 0-10 pts
                        - Technical Signals: 0-10 pts
                        
                        ⚠️ **WARNING:** Short squeezes are extremely volatile. Use proper risk management!
                        """)
                    else:
                        st.warning(f"No candidates found with Squeeze Score ≥ {min_squeeze_score}. Try lowering the threshold.")
                else:
                    progress_bar.empty()
                    st.error("Failed to fetch data. Try again or check your connection.")
        else:
            st.info("🩳 **Scan for short squeeze candidates** — stocks with high short interest, volume surges, and explosive potential.")
            
            st.markdown("""
            ### How Shorties Works
            
            **Scans 120+ known high-short-interest stocks for:**
            
            1. **Short Interest %** — Higher = more fuel for squeeze
            2. **Days to Cover** — Trapped shorts take longer to exit
            3. **Relative Volume** — Volume surge signals squeeze starting
            4. **Price Momentum** — Already moving against shorts
            5. **Volatility** — Higher vol = bigger moves
            6. **Technical Breakout** — Price breaking resistance
            
            **Stock Universe Includes:**
            - 🎮 Meme stocks (GME, AMC, BBBY)
            - 🧬 Heavily shorted biotech
            - ⚡ EV / Clean energy shorts
            - 🪙 Crypto-exposed stocks
            - 📱 Recent IPOs/SPACs
            - 🏪 Retail sector shorts
            
            **Signal Types:**
            - 🚀 **MOON** — Active squeeze, buy signal
            - ⚡ **LOADING** — Building pressure, watch closely
            - 💥 **CRATER** — Reversal, potential short
            - 😴 **DORMANT** — High short % but sleeping
            
            ⚠️ *Short squeezes are HIGH RISK. Position size accordingly!*
            """)

    # ==========================================
    # MODULE I: PICKERS (Ultimate Quant Advisor)
    # ==========================================
    elif mode == "👃 Pickers":
        st.title("👃 Pickers: Ultimate Quant Advisor")
        st.caption("AI-powered stock selection — 5 high-conviction plays you need to own")
        
        st.markdown("### ⚙️ Picker Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            style_pref = st.selectbox("Investment Style", [
                "🎯 Best Overall (Balanced)",
                "📈 Momentum Kings",
                "💎 Value Gems", 
                "⚡ Growth Rockets",
                "🛡️ Quality Compounders",
                "🔄 Mean Reversion"
            ])
        with col2:
            time_horizon = st.selectbox("Time Horizon", ["1-2 Weeks", "1 Month", "3 Months", "6+ Months"], index=1)
        with col3:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
        
        # Elite universe of liquid, optionable stocks (optimized for speed)
        PICKER_UNIVERSE = [
            # Mega-Cap Tech (most liquid)
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX",
            # Financials
            "JPM", "BAC", "GS", "V", "MA", "BLK",
            # Healthcare
            "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE",
            # Consumer
            "WMT", "COST", "HD", "MCD", "NKE", "KO",
            # Industrial & Energy
            "CAT", "DE", "GE", "XOM", "CVX",
            # Growth / High Beta
            "SNOW", "NET", "CRWD", "PANW", "PLTR", "COIN", "MSTR", "DKNG",
        ]
        
        @st.cache_data(ttl=600, show_spinner=False)
        def run_quant_analysis(tickers, style, horizon, risk):
            """Run comprehensive quant analysis on universe"""
            results = []
            
            # Batch download data - smaller batches to avoid timeouts
            price_data = None
            try:
                price_data = yf.download(tickers + ["SPY"], period="1y", progress=False, threads=False)
            except Exception as e:
                st.warning(f"Batch download failed, trying smaller batches...")
                try:
                    # Try in smaller batches
                    price_data = yf.download(tickers[:30] + ["SPY"], period="6mo", progress=False, threads=False)
                except:
                    return pd.DataFrame()
            
            if price_data is None or price_data.empty:
                return pd.DataFrame()
            
            spy_closes = None
            try:
                if isinstance(price_data.columns, pd.MultiIndex):
                    spy_closes = price_data["Close"]["SPY"].dropna()
                else:
                    spy_closes = price_data["Close"].dropna()
            except:
                pass
            
            for ticker in tickers:
                try:
                    # Get price series first (cheaper than API call)
                    if isinstance(price_data.columns, pd.MultiIndex):
                        if ticker not in price_data["Close"].columns:
                            continue
                        closes = price_data["Close"][ticker].dropna()
                        volumes = price_data["Volume"][ticker].dropna()
                    else:
                        closes = price_data["Close"].dropna()
                        volumes = price_data["Volume"].dropna()
                    
                    if len(closes) < 20:  # Reduced requirement
                        continue
                    
                    current_price = closes.iloc[-1]
                    
                    # Get info with timeout protection
                    try:
                        tk = yf.Ticker(ticker)
                        info = tk.info or {}
                    except:
                        info = {}
                    
                    # ========== FUNDAMENTAL SIGNALS ==========
                    # Valuation
                    pe_ratio = info.get("forwardPE") or info.get("trailingPE") or 0
                    ps_ratio = info.get("priceToSalesTrailing12Months") or 0
                    pb_ratio = info.get("priceToBook") or 0
                    peg_ratio = info.get("pegRatio") or 0
                    
                    # Quality
                    roe = (info.get("returnOnEquity") or 0) * 100
                    roa = (info.get("returnOnAssets") or 0) * 100
                    profit_margin = (info.get("profitMargins") or 0) * 100
                    gross_margin = (info.get("grossMargins") or 0) * 100
                    
                    # Growth
                    revenue_growth = (info.get("revenueGrowth") or 0) * 100
                    earnings_growth = (info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth") or 0) * 100
                    
                    # Financial Health
                    debt_to_equity = info.get("debtToEquity") or 0
                    current_ratio = info.get("currentRatio") or 0
                    
                    # ========== TECHNICAL SIGNALS ==========
                    # Momentum (multiple timeframes)
                    mom_5d = ((closes.iloc[-1] / closes.iloc[-5]) - 1) * 100 if len(closes) >= 5 else 0
                    mom_20d = ((closes.iloc[-1] / closes.iloc[-20]) - 1) * 100 if len(closes) >= 20 else 0
                    mom_60d = ((closes.iloc[-1] / closes.iloc[-60]) - 1) * 100 if len(closes) >= 60 else 0
                    mom_250d = ((closes.iloc[-1] / closes.iloc[-250]) - 1) * 100 if len(closes) >= 250 else mom_60d
                    
                    # Trend Strength
                    sma_20 = closes.rolling(20).mean().iloc[-1]
                    sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else sma_20
                    sma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else sma_50
                    
                    above_20 = current_price > sma_20
                    above_50 = current_price > sma_50
                    above_200 = current_price > sma_200
                    golden_cross = sma_50 > sma_200
                    
                    # Volatility
                    returns = closes.pct_change().dropna()
                    vol_20 = returns.tail(20).std() * np.sqrt(252) * 100
                    vol_60 = returns.tail(60).std() * np.sqrt(252) * 100
                    
                    # Volume Analysis
                    avg_vol_20 = volumes.tail(20).mean()
                    rel_volume = volumes.iloc[-1] / avg_vol_20 if avg_vol_20 > 0 else 1
                    vol_trend = volumes.tail(5).mean() / volumes.tail(20).mean() if volumes.tail(20).mean() > 0 else 1
                    
                    # Relative Strength vs SPY
                    if spy_closes is not None and len(spy_closes) >= 60:
                        spy_mom = ((spy_closes.iloc[-1] / spy_closes.iloc[-60]) - 1) * 100
                        rs_vs_spy = mom_60d - spy_mom
                    else:
                        rs_vs_spy = 0
                    
                    # Mean Reversion Signal
                    zscore = (current_price - sma_20) / (closes.rolling(20).std().iloc[-1] + 0.001)
                    
                    # RSI
                    delta = closes.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain.iloc[-1] / (loss.iloc[-1] + 0.001)
                    rsi = 100 - (100 / (1 + rs))
                    
                    # 52-week position
                    high_52w = closes.max()
                    low_52w = closes.min()
                    pct_from_high = ((current_price / high_52w) - 1) * 100
                    pct_from_low = ((current_price / low_52w) - 1) * 100
                    
                    # ========== COMPOSITE SCORING ==========
                    
                    # Value Score (0-100)
                    value_score = 0
                    if 0 < pe_ratio < 15:
                        value_score += 25
                    elif 15 <= pe_ratio < 25:
                        value_score += 15
                    elif 25 <= pe_ratio < 40:
                        value_score += 5
                    
                    if 0 < ps_ratio < 2:
                        value_score += 25
                    elif 2 <= ps_ratio < 5:
                        value_score += 15
                    elif 5 <= ps_ratio < 10:
                        value_score += 5
                    
                    if 0 < pb_ratio < 2:
                        value_score += 25
                    elif 2 <= pb_ratio < 4:
                        value_score += 15
                    
                    if 0 < peg_ratio < 1:
                        value_score += 25
                    elif 1 <= peg_ratio < 2:
                        value_score += 15
                    elif 2 <= peg_ratio < 3:
                        value_score += 5
                    
                    # Momentum Score (0-100)
                    momentum_score = 0
                    if mom_20d > 10:
                        momentum_score += 25
                    elif mom_20d > 5:
                        momentum_score += 15
                    elif mom_20d > 0:
                        momentum_score += 8
                    
                    if mom_60d > 20:
                        momentum_score += 25
                    elif mom_60d > 10:
                        momentum_score += 15
                    elif mom_60d > 0:
                        momentum_score += 8
                    
                    if rs_vs_spy > 10:
                        momentum_score += 25
                    elif rs_vs_spy > 5:
                        momentum_score += 15
                    elif rs_vs_spy > 0:
                        momentum_score += 8
                    
                    if golden_cross and above_50:
                        momentum_score += 15
                    if rel_volume > 1.5:
                        momentum_score += 10
                    
                    # Quality Score (0-100)
                    quality_score = 0
                    if roe > 20:
                        quality_score += 25
                    elif roe > 15:
                        quality_score += 18
                    elif roe > 10:
                        quality_score += 10
                    
                    if profit_margin > 20:
                        quality_score += 25
                    elif profit_margin > 10:
                        quality_score += 15
                    elif profit_margin > 5:
                        quality_score += 8
                    
                    if gross_margin > 50:
                        quality_score += 20
                    elif gross_margin > 30:
                        quality_score += 12
                    
                    if debt_to_equity < 50:
                        quality_score += 15
                    elif debt_to_equity < 100:
                        quality_score += 8
                    
                    if current_ratio > 2:
                        quality_score += 15
                    elif current_ratio > 1.5:
                        quality_score += 10
                    
                    # Growth Score (0-100)
                    growth_score = 0
                    if revenue_growth > 30:
                        growth_score += 35
                    elif revenue_growth > 20:
                        growth_score += 25
                    elif revenue_growth > 10:
                        growth_score += 15
                    elif revenue_growth > 0:
                        growth_score += 5
                    
                    if earnings_growth > 30:
                        growth_score += 35
                    elif earnings_growth > 20:
                        growth_score += 25
                    elif earnings_growth > 10:
                        growth_score += 15
                    elif earnings_growth > 0:
                        growth_score += 5
                    
                    if pct_from_high > -10:  # Near highs = growth mode
                        growth_score += 20
                    elif pct_from_high > -20:
                        growth_score += 10
                    
                    # Mean Reversion Score (0-100)
                    reversion_score = 0
                    if zscore < -2:
                        reversion_score = 90  # Extremely oversold
                    elif zscore < -1.5:
                        reversion_score = 75
                    elif zscore < -1:
                        reversion_score = 60
                    elif zscore > 2:
                        reversion_score = 20  # Overbought (penalize)
                    else:
                        reversion_score = 40  # Neutral
                    
                    if rsi < 30:
                        reversion_score += 10
                    elif rsi > 70:
                        reversion_score -= 10
                    
                    # ========== STYLE-WEIGHTED ALPHA SCORE ==========
                    if "Momentum" in style:
                        alpha = momentum_score * 0.50 + growth_score * 0.25 + quality_score * 0.15 + value_score * 0.10
                    elif "Value" in style:
                        alpha = value_score * 0.45 + quality_score * 0.30 + reversion_score * 0.15 + momentum_score * 0.10
                    elif "Growth" in style:
                        alpha = growth_score * 0.45 + momentum_score * 0.30 + quality_score * 0.15 + value_score * 0.10
                    elif "Quality" in style:
                        alpha = quality_score * 0.50 + value_score * 0.25 + growth_score * 0.15 + momentum_score * 0.10
                    elif "Reversion" in style:
                        alpha = reversion_score * 0.50 + value_score * 0.25 + quality_score * 0.15 + momentum_score * 0.10
                    else:  # Balanced
                        alpha = (momentum_score * 0.25 + quality_score * 0.25 + value_score * 0.20 + 
                                growth_score * 0.20 + reversion_score * 0.10)
                    
                    # Risk adjustment
                    if risk == "Conservative":
                        if vol_20 > 50:
                            alpha *= 0.8  # Penalize high vol
                        if debt_to_equity > 100:
                            alpha *= 0.9
                    elif risk == "Aggressive":
                        if vol_20 > 40:
                            alpha *= 1.1  # Reward vol for aggro
                        if momentum_score > 60:
                            alpha *= 1.1
                    
                    # Determine play type
                    scores = {
                        "Momentum": momentum_score,
                        "Value": value_score,
                        "Quality": quality_score,
                        "Growth": growth_score,
                        "Reversion": reversion_score
                    }
                    primary_style = max(scores, key=scores.get)
                    
                    # Generate thesis
                    thesis_parts = []
                    if momentum_score > 60:
                        thesis_parts.append(f"Strong momentum (+{mom_20d:.0f}% 20d)")
                    if rs_vs_spy > 5:
                        thesis_parts.append(f"Outperforming SPY by {rs_vs_spy:.0f}%")
                    if value_score > 60:
                        thesis_parts.append(f"Cheap valuation (P/E {pe_ratio:.0f})")
                    if quality_score > 60:
                        thesis_parts.append(f"High quality (ROE {roe:.0f}%)")
                    if growth_score > 60:
                        thesis_parts.append(f"Growing fast ({revenue_growth:.0f}% rev)")
                    if zscore < -1.5:
                        thesis_parts.append(f"Oversold (z={zscore:.1f})")
                    if golden_cross:
                        thesis_parts.append("Golden cross active")
                    if rel_volume > 2:
                        thesis_parts.append(f"Volume surge ({rel_volume:.1f}x)")
                    
                    thesis = " | ".join(thesis_parts[:3]) if thesis_parts else "Balanced metrics"
                    
                    # Entry strategy
                    if above_20 and mom_5d > 0:
                        entry = "Buy on pullback to 20-SMA"
                    elif zscore < -1:
                        entry = "Buy now — oversold bounce"
                    elif rel_volume > 1.5 and mom_5d > 2:
                        entry = "Buy breakout with volume"
                    else:
                        entry = "Scale in on weakness"
                    
                    # Risk factors
                    risks = []
                    if vol_20 > 50:
                        risks.append("High volatility")
                    if pct_from_high < -30:
                        risks.append("Significant drawdown")
                    if debt_to_equity > 100:
                        risks.append("High debt")
                    if rsi > 70:
                        risks.append("Overbought short-term")
                    risk_str = ", ".join(risks[:2]) if risks else "Standard market risk"
                    
                    results.append({
                        "Ticker": ticker,
                        "Alpha": alpha,
                        "Style": primary_style,
                        "Price": current_price,
                        "Mom20": momentum_score,
                        "Quality": quality_score,
                        "Value": value_score,
                        "Growth": growth_score,
                        "RS_SPY": rs_vs_spy,
                        "Vol": vol_20,
                        "RSI": rsi,
                        "Thesis": thesis,
                        "Entry": entry,
                        "Risks": risk_str,
                        "PE": pe_ratio,
                        "ROE": roe,
                        "RevGr": revenue_growth,
                        "FromHigh": pct_from_high
                    })
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(results)
        
        # Run Analysis
        if st.button("👃 Find My 5 Best Picks", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing quant analysis...")
            
            with st.spinner("Running comprehensive multi-factor analysis..."):
                progress_bar.progress(30, text="Analyzing fundamentals...")
                
                df_picks = run_quant_analysis(
                    PICKER_UNIVERSE,
                    style_pref,
                    time_horizon,
                    risk_tolerance
                )
                
                progress_bar.progress(80, text="Ranking opportunities...")
                
                if df_picks is not None and not df_picks.empty:
                    # Get top 5
                    df_picks = df_picks.sort_values("Alpha", ascending=False)
                    top_5 = df_picks.head(5)
                    
                    progress_bar.progress(100, text="Analysis complete!")
                    progress_bar.empty()
                    
                    st.markdown("---")
                    st.markdown(f"## 🏆 Your Top 5 Picks — {style_pref}")
                    st.caption(f"Time Horizon: {time_horizon} | Risk: {risk_tolerance}")
                    
                    # Display each pick with full detail
                    for i, (_, pick) in enumerate(top_5.iterrows(), 1):
                        # Color based on style
                        style_colors = {
                            "Momentum": "#10B981",
                            "Value": "#3B82F6",
                            "Quality": "#8B5CF6",
                            "Growth": "#F59E0B",
                            "Reversion": "#EF4444"
                        }
                        color = style_colors.get(pick["Style"], "#6B7280")
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                                    padding: 20px; border-radius: 16px; border-left: 4px solid {color}; 
                                    margin: 16px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 2rem; font-weight: 800;">#{i}</span>
                                    <span style="font-size: 1.8rem; font-weight: 700; margin-left: 12px;">{pick["Ticker"]}</span>
                                    <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 20px; 
                                                 font-size: 0.8rem; margin-left: 12px;">{pick["Style"]}</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.5rem; font-weight: 700;">${pick["Price"]:.2f}</div>
                                    <div style="font-size: 0.9rem; color: {color};">Alpha: {pick["Alpha"]:.0f}</div>
                                </div>
                            </div>
                            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid {color}40;">
                                <div style="font-weight: 600; color: {color};">📊 Why This Pick:</div>
                                <div style="color: #CBD5E1; margin-top: 4px;">{pick["Thesis"]}</div>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
                                <div>
                                    <div style="font-weight: 600; color: #10B981;">🎯 Entry Strategy:</div>
                                    <div style="color: #CBD5E1;">{pick["Entry"]}</div>
                                </div>
                                <div>
                                    <div style="font-weight: 600; color: #EF4444;">⚠️ Key Risks:</div>
                                    <div style="color: #CBD5E1;">{pick["Risks"]}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics comparison
                    st.subheader("📊 Picks Comparison")
                    
                    comparison_df = top_5[["Ticker", "Alpha", "Style", "Price", "Quality", "Value", "Growth", "RS_SPY", "Vol", "RSI", "PE", "ROE", "RevGr", "FromHigh"]]
                    
                    st.dataframe(
                        comparison_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Alpha": st.column_config.ProgressColumn("Alpha", format="%d", min_value=0, max_value=100),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "Quality": st.column_config.NumberColumn("Quality", format="%d"),
                            "Value": st.column_config.NumberColumn("Value", format="%d"),
                            "Growth": st.column_config.NumberColumn("Growth", format="%d"),
                            "RS_SPY": st.column_config.NumberColumn("vs SPY", format="%.1f%%"),
                            "Vol": st.column_config.NumberColumn("Vol%", format="%.0f%%"),
                            "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                            "PE": st.column_config.NumberColumn("P/E", format="%.1f"),
                            "ROE": st.column_config.NumberColumn("ROE%", format="%.1f%%"),
                            "RevGr": st.column_config.NumberColumn("Rev Gr", format="%.1f%%"),
                            "FromHigh": st.column_config.NumberColumn("vs 52H", format="%.1f%%")
                        }
                    )
                    
                    # Full rankings
                    with st.expander("📋 Full Rankings (All Analyzed Stocks)"):
                        st.dataframe(
                            df_picks[["Ticker", "Alpha", "Style", "Price", "Quality", "Value", "Growth", "Thesis"]].head(30),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Alpha": st.column_config.ProgressColumn("Alpha", format="%d", min_value=0, max_value=100),
                                "Price": st.column_config.NumberColumn("Price", format="$%.2f")
                            }
                        )
                    
                    # Style breakdown
                    st.markdown("---")
                    st.subheader("📈 Play Type Breakdown")
                    
                    style_counts = df_picks["Style"].value_counts()
                    cols = st.columns(5)
                    for i, (style, count) in enumerate(style_counts.items()):
                        with cols[i % 5]:
                            emoji = {"Momentum": "📈", "Value": "💎", "Quality": "🛡️", "Growth": "⚡", "Reversion": "🔄"}.get(style, "📊")
                            st.metric(f"{emoji} {style}", count)
                    
                    st.markdown("---")
                    st.markdown("""
                    ### 🔑 Pickers Methodology
                    
                    **Multi-Factor Alpha Score combines:**
                    
                    | Factor | Components |
                    |--------|------------|
                    | **Momentum** | 20/60/250-day returns, RS vs SPY, trend strength, volume |
                    | **Value** | P/E, P/S, P/B, PEG ratio |
                    | **Quality** | ROE, profit margin, gross margin, debt levels |
                    | **Growth** | Revenue growth, earnings growth, near-highs |
                    | **Reversion** | Z-score, RSI, distance from moving averages |
                    
                    **Style Weightings:**
                    - 🎯 Balanced: Equal weight all factors
                    - 📈 Momentum: 50% mom, 25% growth, 15% quality, 10% value
                    - 💎 Value: 45% value, 30% quality, 15% reversion, 10% mom
                    - ⚡ Growth: 45% growth, 30% mom, 15% quality, 10% value
                    - 🛡️ Quality: 50% quality, 25% value, 15% growth, 10% mom
                    - 🔄 Reversion: 50% reversion, 25% value, 15% quality, 10% mom
                    """)
                    
                else:
                    progress_bar.empty()
                    st.warning("⚠️ Analysis returned no results. This can happen due to API rate limits.")
                    st.info("**Try again in a few seconds** — yfinance sometimes throttles requests on Streamlit Cloud.")
                    
                    # Offer a quick retry with fewer stocks
                    if st.button("🔄 Quick Retry (Top 10 Stocks Only)"):
                        quick_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "UNH"]
                        with st.spinner("Retrying with smaller set..."):
                            df_retry = run_quant_analysis(quick_tickers, style_pref, time_horizon, risk_tolerance)
                            if df_retry is not None and not df_retry.empty:
                                st.success(f"Found {len(df_retry)} stocks!")
                                st.dataframe(df_retry[["Ticker", "Alpha", "Style", "Price", "Thesis"]].head(5))
        else:
            st.info("👃 **Click above to find your 5 best picks** based on comprehensive quant analysis")
            
            st.markdown("""
            ### How Pickers Works
            
            **Analyzes 80+ elite liquid stocks across 5 dimensions:**
            
            📈 **Momentum** — Price momentum, relative strength, trend, volume
            
            💎 **Value** — P/E, P/S, P/B, PEG ratios vs sector
            
            🛡️ **Quality** — ROE, margins, debt levels, financial health
            
            ⚡ **Growth** — Revenue & earnings growth rates
            
            🔄 **Reversion** — Oversold bounces, mean reversion setups
            
            **For Each Pick You Get:**
            - 🎯 **Alpha Score** — Composite ranking (0-100)
            - 📊 **Play Type** — What style of opportunity it is
            - 💡 **Thesis** — Why this stock, what's the edge
            - 🚀 **Entry Strategy** — How to get in
            - ⚠️ **Key Risks** — What could go wrong
            
            *Select your style preference and risk tolerance, then let the quant engine find your winners.*
            """)


if __name__ == "__main__":
    main()
