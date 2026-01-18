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

# Optional Finviz dependency for Ryan Model 2.0
try:
    from finvizfinance.screener.overview import Overview as FinvizOverview
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False

# Optional statsmodels for Pairs cointegration
try:
    from statsmodels.tsa.stattools import coint
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


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
        
        .stApp { background-color: #F1F5F9; color: #0F172A; font-family: 'Inter', sans-serif; }
        section[data-testid="stSidebar"] { 
            background-color: #1E293B !important; 
            border-right: 1px solid #334155; 
        }
        
        section[data-testid="stSidebar"] * {
            color: #F8FAFC !important;
        }
        
        /* METRIC CARDS (PIVOT: DARK WITH WHITE TEXT) */
        div[data-testid="stMetric"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            padding: 15px;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }
        div[data-testid="stMetricValue"] { 
            color: #FFFFFF !important; 
            font-family: 'JetBrains Mono', monospace; 
            font-weight: 700; 
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * { 
            color: #FFFFFF !important; 
            font-size: 0.85rem !important; 
            letter-spacing: 0.5px; 
            font-weight: 600; 
            text-transform: uppercase !important; 
        }
        
        /* Metric label inner elements - force white */
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] span,
        div[data-testid="stMetricLabel"] div,
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p,
        div[data-testid="stMetric"] label span {
            color: #FFFFFF !important;
        }
        div[data-testid="stMetricDelta"] > div {
            color: #FFFFFF !important;
        }
        
        /* ALERTS */
        .signal-box {
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-family: 'JetBrains Mono', monospace;
            border-left: 4px solid #64748B;
            background: #FFFFFF;
        }
        
        /* EXPANDER CONTENT - ENSURE READABLE TEXT */
        .streamlit-expanderContent {
            color: #0F172A !important;
        }
        
        .streamlit-expanderContent div, 
        .streamlit-expanderContent span,
        .streamlit-expanderContent p,
        .streamlit-expanderContent label {
            color: #0F172A !important;
        }
        
        /* SLIDER LABELS - ENSURE VISIBILITY */
        .stSlider > div > div > div > div {
            color: #0F172A !important;
        }
        
        .stSlider label {
            color: #0F172A !important;
        }
        
        /* PRIMARY BUTTON - WHITE TEXT ON DARK BACKGROUND */
        button[kind="primary"], 
        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            color: #FFFFFF !important;
            background-color: #1E293B !important;
            border: none !important;
        }
        
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #334155 !important;
            color: #FFFFFF !important;
        }
        
        /* ALL BUTTONS - WHITE TEXT (comprehensive selectors) */
        .stButton > button,
        .stButton button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-tertiary"],
        button[data-testid="stBaseButton-secondary"],
        button[kind="secondary"],
        div.stButton > button,
        div[data-testid="stButton"] > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            color: #FFFFFF !important;
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
        }
        
        /* Button hover states */
        .stButton > button:hover,
        .stButton button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        div.stButton > button:hover {
            color: #FFFFFF !important;
            background-color: #475569 !important;
        }
        
        /* Button text spans inside buttons */
        .stButton > button span,
        .stButton > button p,
        .stButton > button div,
        button span,
        button p {
            color: #FFFFFF !important;
        }
        
        /* GLOBAL READABILITY IMPROVEMENTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 15px !important;
        }
        
        /* SIDEBAR - MAKE LABELS READABLE */
        section[data-testid="stSidebar"] .stRadio > label {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #F8FAFC !important;
            padding: 12px 16px !important;
            background: #334155 !important;
            border-radius: 8px !important;
            margin: 4px 0 !important;
            display: block !important;
            border: 1px solid #475569 !important;
            transition: all 0.2s !important;
        }
        
        section[data-testid="stSidebar"] .stRadio > label:hover {
            background: #475569 !important;
            border-color: #3B82F6 !important;
        }
        
        /* SIDEBAR - SELECTED/ACTIVE MODULE INDICATOR */
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
        section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked),
        section[data-testid="stSidebar"] .stRadio div[data-checked="true"] {
            background: #3B82F6 !important;
            border-color: #60A5FA !important;
            border-left: 4px solid #60A5FA !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        
        /* Alternative selector for checked radio */
        section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div,
        section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked ~ div {
            background: #3B82F6 !important;
        }
        
        /* Hide default radio circles in sidebar */
        section[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
            display: none !important;
        }
        
        /* HEADINGS - PURE BLACK */
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
            font-weight: 700 !important;
        }
        
        h1 { font-size: 2.25rem !important; }
        h2 { font-size: 1.875rem !important; }
        h3 { font-size: 1.5rem !important; }
        
        /* Ensure all text in main container is high-contrast dark on the light background */
        p, li, span, div, [class*="css"], .stCaption {
            color: #0F172A !important;
        }
        
        /* TAB LABELS - DARK TEXT ON LIGHT BACKGROUND */
        .stTabs [data-baseweb="tab-list"] button,
        .stTabs [data-baseweb="tab-list"] button p,
        .stTabs [data-baseweb="tab-list"] button div,
        .stTabs [role="tablist"] button,
        button[role="tab"],
        button[role="tab"] p,
        button[role="tab"] span {
            color: #0F172A !important;
            background-color: transparent !important;
        }
        
        /* Active tab styling */
        .stTabs [aria-selected="true"],
        button[role="tab"][aria-selected="true"] {
            color: #1E40AF !important;
            font-weight: 600 !important;
            border-bottom-color: #3B82F6 !important;
        }

        /* Support for metric delta contrast */
        div[data-testid="stMetricDelta"] > div {
            color: #FFFFFF !important;
        }
        
        div[data-testid="stMetricDelta"] {
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            color: #000000 !important;
        }

        /* Ensure all text in main container is black */
        .main .block-container p, 
        .main .block-container li, 
        .main .block-container span, 
        .main .block-container div {
            color: #000000 !important;
        }
        
        /* INPUT FIELDS - BLACK LABELS */
        label {
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        
        input, textarea, select {
            font-size: 1rem !important;
            color: #000000 !important;
        }
        
        /* BUTTONS - HIGH CONTRAST */
        button {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        
        /* DATAFRAME/TABLE - READABLE */
        .dataframe {
            font-size: 0.9rem !important;
        }
        
        .dataframe th {
            background-color: #1F2937 !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 12px !important;
        }
        
        .dataframe td {
            padding: 10px !important;
            color: #374151 !important;
        }
        
        /* CAPTION - MORE VISIBLE */
        .caption, [data-testid="stCaptionContainer"] {
            color: #6B7280 !important;
            font-size: 0.9rem !important;
        }
        
        /* EXPANDER - READABLE */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #111827 !important;
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
            return "NEUTRAL", 0.0, pd.DataFrame(), {}

        factors = pd.DataFrame(index=closes.index)
        needed = set(self.MACRO_TICKERS)
        existing = set(closes.columns)
        if not needed.issubset(existing):
            return "NEUTRAL", 0.0, pd.DataFrame(), {}

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
        
        # Extract current ratios and z-scores
        current_ratios = {
            'Credit': {'value': factors['Credit'].iloc[-1], 'z': z_scores['Credit'].iloc[-1]},
            'Consumer': {'value': factors['Consumer'].iloc[-1], 'z': z_scores['Consumer'].iloc[-1]},
            'Growth': {'value': factors['Growth'].iloc[-1], 'z': z_scores['Growth'].iloc[-1]},
            'Fear': {'value': factors['Fear'].iloc[-1], 'z': z_scores['Fear'].iloc[-1]}
        }
        
        return regime, current_score, z_scores, current_ratios

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

    def get_beta(self, benchmark_ticker="SPY", period="2y"):
        """Calculate Beta vs Benchmark"""
        try:
            # We need 2 years of data for a solid beta
            # Note: history_df passed in __init__ might be shorter (1y), 
            # so we might need to fetch more if strictly required, 
            # but for now we'll use what we have or fetch if needed.
            
            # Use the hydra cache or direct fetch if needed
            # For simplicity in this context, we'll try to use the existing history if sufficient
            # or do a quick fetch.
            
            if self.history.empty:
                return 1.0
            
            # Align with benchmark
            # This requires fetching benchmark data. 
            # We will use yf.download for simplest implementation here.
            # In a loop this is slow, so ideally we pass benchmark data in.
            
            # OPTIMIZATION: Assume benchmark data is passed or we assume beta=1 for now 
            # if we want to avoid network calls inside the class methods.
            
            # However, for the "Beta Brake" we need it.
            # Let's implementation a simple logic: 
            # Calculate beta based on the shared timeframe of self.history vs SPY.
            
            spy = yf.download(benchmark_ticker, period="1y", progress=False)
            if spy.empty:
                return 1.0
                
            # Prepare dataframes
            asset_dates = self.history.index
            spy = spy.reindex(asset_dates).ffill().dropna()
            asset = self.history.reindex(spy.index).dropna()
            
            if len(asset) < 30:
                return 1.0
                
            rets_asset = asset['Close'].pct_change().dropna()
            rets_spy = spy['Close'].pct_change().dropna()
            
            # Align again after pct_change
            common_idx = rets_asset.index.intersection(rets_spy.index)
            rets_asset = rets_asset.loc[common_idx]
            rets_spy = rets_spy.loc[common_idx]
            
            if len(rets_asset) < 30:
                return 1.0
                
            covariance = np.cov(rets_asset, rets_spy)[0][1]
            variance = np.var(rets_spy)
            
            beta = covariance / variance
            return float(beta)
        except:
            return 1.0

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

    def run_layer4_jump_diffusion(self):
        """Layer 4: Tail Risk (Simplified Merton Jump Diffusion)"""
        if self.history.empty or len(self.history) < 60:
            return "UNKNOWN", 0.0, 0.0
        
        returns = self.history['Close'].pct_change().dropna()
        
        # Calculate extreme move threshold (3 standard deviations)
        mu = returns.mean()
        sigma = returns.std()
        threshold = 3 * sigma
        
        # Count jumps (moves > 3 sigma)
        jumps = returns[abs(returns - mu) > threshold]
        jump_count = len(jumps)
        total_days = len(returns)
        
        # Jump intensity (lambda): probability of jump per day
        jump_intensity = jump_count / total_days if total_days > 0 else 0
        
        # Average jump size
        jump_size = abs(jumps).mean() if len(jumps) > 0 else 0
        
        # Tail risk score (0-1, lower is better)
        # Based on: how often jumps occur + how big they are
        tail_risk_score = min(1.0, (jump_intensity * 10) + (jump_size * 2))
        
        # Classification
        if tail_risk_score < 0.3:
            risk_class = "LOW"
        elif tail_risk_score < 0.6:
            risk_class = "MEDIUM"
        else:
            risk_class = "HIGH"
        
        # Probability of extreme move in next period (simplified)
        jump_prob = min(0.99, jump_intensity * 100)  # Convert to percentage
        
        return risk_class, tail_risk_score, jump_prob


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
        # Added Large/Mega Caps (25 more)
        "NFLX",
        "BKNG",
        "ABNB",
        "UBER",
        "SHOP",
        "NOW",
        "PANW",
        "SNOW",
        "CRWD",
        "ZS",
        "INTC",
        "MU",
        "AMAT",
        "LRCX",
        "KLAC",
        "TMO",
        "DHR",
        "ABT",
        "ISRG",
        "BMY",
        "VRTX",
        "REGN",
        "LOW",
        "TGT",
        "CMG",
        "LULU",
        "YUM",
        "EL",
        "CL",
        # Defense & Aerospace
        "NOC",
        "GD",
        "LHX",
        "HII",
        # Housing / Homebuilders
        "DHI",
        "LEN",
        "PHM",
        "NVR",
        "TOL",
        "KBH",
        # Energy / Materials / Utilities
        "FCX",
        "NEM",
        "FSLR",
        "ENPH",
        "NEE",
        # User Custom Adds
        "TXT",
        "TDY",
        "ENVX",
        "RCAT",
        "UMAC",
        "DPRO",
        "AVAV",
        "ONDS",
        "ARM",
        "NBIS",
        "SOFI",
        "OSCR",
        "EOSE",
        "CORZ",
        "CRCL",
        "CVNA",
        "SMR",
        "KTOS",
        "FOUR",
        "NVO",
        "DLO",
        "NXXT",
        "RKLB",
        "ASTS",
        "HIMS",
        "IREN",
        "ANF",
        "MSTR",
    ]

    @staticmethod
    def calculate_flow_signal(ticker, trend, momentum):
        """
        Calculate options flow signal using VOL/OI and Put/Call ratios.
        Returns: (signal_label, flow_score, details)
        
        Signal Labels:
        - 🟢 BULLISH: Strong call flow, VOL/OI confirms fresh buying
        - 🟡 NEUTRAL: Mixed signals or insufficient data
        - 🔴 BEARISH: Strong put flow, VOL/OI confirms fresh selling
        - ⚠️ FAKE: Flow contradicts price action (likely hedging/closing)
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get options expirations
            expirations = stock.options
            if not expirations or len(expirations) == 0:
                return "🟡 NEUTRAL", 0, "No opts"
            
            # Focus on near-term expirations (first 2)
            total_call_vol = 0
            total_put_vol = 0
            total_call_oi = 0
            total_put_oi = 0
            
            for exp in expirations[:2]:
                try:
                    chain = stock.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts
                    
                    # Sum volumes and OI
                    if 'volume' in calls.columns:
                        total_call_vol += calls['volume'].fillna(0).sum()
                    if 'volume' in puts.columns:
                        total_put_vol += puts['volume'].fillna(0).sum()
                    if 'openInterest' in calls.columns:
                        total_call_oi += calls['openInterest'].fillna(0).sum()
                    if 'openInterest' in puts.columns:
                        total_put_oi += puts['openInterest'].fillna(0).sum()
                except:
                    continue
            
            # Avoid division by zero
            if total_call_vol == 0 and total_put_vol == 0:
                return "🟡 NEUTRAL", 0, "No vol"
            
            # Calculate key ratios
            pc_ratio = total_put_vol / max(total_call_vol, 1)
            call_vol_oi = total_call_vol / max(total_call_oi, 1)
            put_vol_oi = total_put_vol / max(total_put_oi, 1)
            
            # --- SCORING ENGINE ---
            flow_score = 0
            
            # 1. Put/Call Ratio Sentiment (±40 points)
            if pc_ratio < 0.5:  # Very call heavy
                flow_score += 40
            elif pc_ratio < 0.7:
                flow_score += 25
            elif pc_ratio < 0.9:
                flow_score += 10
            elif pc_ratio > 2.0:  # Very put heavy
                flow_score -= 40
            elif pc_ratio > 1.5:
                flow_score -= 25
            elif pc_ratio > 1.2:
                flow_score -= 10
            
            # 2. VOL/OI Ratio - Fresh Money Indicator (±30 points)
            # High VOL/OI = new positions opening
            if call_vol_oi > 10:  # Very high call activity
                flow_score += 30
            elif call_vol_oi > 5:
                flow_score += 15
            elif call_vol_oi > 2:
                flow_score += 5
            
            if put_vol_oi > 10:  # Very high put activity
                flow_score -= 30
            elif put_vol_oi > 5:
                flow_score -= 15
            elif put_vol_oi > 2:
                flow_score -= 5
            
            # 3. Determine base signal
            if flow_score >= 35:
                base_signal = "BULLISH"
            elif flow_score <= -35:
                base_signal = "BEARISH"
            else:
                base_signal = "NEUTRAL"
            
            # 4. FAKE Detection - Flow contradicts price action
            # If bullish flow but bearish trend/momentum = likely hedging
            # If bearish flow but bullish trend/momentum = likely hedging
            is_trend_bullish = (trend == "BULLISH" and momentum == "BULLISH")
            is_trend_bearish = (trend == "BEARISH" and momentum == "BEARISH")
            
            if base_signal == "BULLISH" and is_trend_bearish:
                return "⚠️ FAKE", flow_score, f"P/C:{pc_ratio:.1f} vs 📉"
            elif base_signal == "BEARISH" and is_trend_bullish:
                return "⚠️ FAKE", flow_score, f"P/C:{pc_ratio:.1f} vs 📈"
            
            # 5. Return final signal
            details = f"P/C:{pc_ratio:.1f}"
            if base_signal == "BULLISH":
                return "🟢 BULLISH", flow_score, details
            elif base_signal == "BEARISH":
                return "🔴 BEARISH", flow_score, details
            else:
                return "🟡 NEUTRAL", flow_score, details
                
        except Exception:
            return "🟡 NEUTRAL", 0, "Error"

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

                # --- QUANT SANDWICH (3-Layer Stack) ---
                quant_score = 0
                regime = "NEUTRAL"
                vol_mult = 1.0
                tail_risk = "MEDIUM"
                
                try:
                    # Run Quant Stack Engine
                    qs = QuantStackEngine(symbol, df)
                    
                    # Layer 1: Four Horsemen Macro Regime
                    regime, macro_score, _, horsemen = qs.run_layer1_regime()
                    
                    # Layer 2: GARCH Volatility Sizing
                    vol_mult, vol_ratio, _ = qs.run_layer3_garch()
                    
                    # Layer 3: Jump Diffusion Tail Risk
                    tail_risk, tail_score, jump_prob = qs.run_layer4_jump_diffusion()
                    
                    # Composite Quant Score (0-100)
                    # Layer 1: Regime (30 points)
                    if regime == "BULL":
                        quant_score += 30
                    elif regime == "NEUTRAL":
                        quant_score += 15
                    # BEAR = 0 points
                    
                    # Layer 2: Volatility Sizing (30 points)
                    # Reward stable/low vol, penalize high vol
                    if vol_mult >= 1.0:  # Low vol environment
                        quant_score += 30
                    elif vol_mult >= 0.7:
                        quant_score += 20
                    else:  # High vol
                        quant_score += 10
                    
                    # Layer 3: Tail Risk (40 points)
                    # Reward low tail risk
                    if tail_risk == "LOW":
                        quant_score += 40
                    elif tail_risk == "MEDIUM":
                        quant_score += 25
                    else:  # HIGH
                        quant_score += 10
                    
                except Exception:
                    # If Quant Stack fails, default to neutral scores
                    quant_score = 50
                    regime = "NEUTRAL"
                    vol_mult = 1.0
                    tail_risk = "UNKNOWN"

                # --- OPTIONS FLOW SIGNAL ---
                try:
                    flow_signal, flow_score, flow_details = ScannerEngine.calculate_flow_signal(symbol, trend, mom)
                except Exception:
                    flow_signal = "🟡 NEUTRAL"
                    flow_details = "Error"

                results.append(
                    {
                        "Ticker": symbol,
                        "Action": action,
                        "Flow": flow_signal,  # NEW: Options flow signal
                        "Confidence": confidence,
                        "Quant_Score": quant_score,
                        "Regime": regime,
                        "Vol_Sizing": f"{vol_mult:.1f}x",
                        "Tail_Risk": tail_risk,
                        "Price": curr_p,
                        "Trend": trend,
                        "Momentum": mom,
                        "Squeeze": "COILED" if is_squeeze else "LOOSE",
                        "Health": health,
                        "Bandwidth": current_bw,
                        "Vol_Vel": vol_status,
                        "Flow_Details": flow_details,  # NEW: P/C ratio details
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
        
        mode = st.radio("Select Module", ["📺 Dashboard", "🎯 Sniper", "🦅 Hunter", "🧠 Ryan Model 2.0", "🔗 Pairs"], label_visibility="collapsed")
        
        st.markdown("---")
        st.caption("Tap ✕ or swipe to close")

    # --- SNIPER MODE ---
    if mode == "🎯 Sniper":
        st.title("🎯 Sniper: Single Stock Analysis")
        st.caption("Volatility Squeeze + Volume Velocity + Relative Strength + Monte Carlo")
        
        # Ticker input in main area (SINGLE INPUT ONLY)
        ticker = st.text_input("Enter Ticker Symbol", value="TSLA", placeholder="e.g., TSLA, AAPL, NVDA").upper()
        
        # Use default values for calculations (hidden from user)
        rf, cap, kelly = 4.5, 100000, 0.5
        
        st.markdown("---")
        
        # Fetch data
        hydra = HydraEngine()
        with st.spinner(f"Connecting to {ticker}..."):
            spot, src = hydra.get_spot(ticker)
            hist = hydra.get_history(ticker)

        if spot is None:
            st.error(f"❌ **Unable to fetch data for ticker: {ticker}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.warning("""
                **Possible Reasons:**
                - Ticker doesn't exist or is delisted
                - No market data available
                - Network/API issue
                """)
            
            with col2:
                st.info("""
                **Try These Tickers:**
                - **Tech**: AAPL, MSFT, NVDA, TSLA
                - **ETFs**: SPY, QQQ, DIA, IWM
                - **Other**: META, AMZN, GOOGL, AMD
                """)
            
            with st.expander("🔧 Technical Debug Info"):
                st.code(hydra.get_debug_info())
                st.caption("💡 **Tip**: Run `pip install --upgrade yfinance` in your terminal")
            return
        if hist is None or hist.empty:
            st.warning("⚠️ Limited historical data available. Some features may be restricted.")
            # Create minimal DataFrame with just the current price
            hist = pd.DataFrame({'Close': [spot], 'Volume': [0], 'High': [spot], 'Low': [spot]}, index=[pd.Timestamp.now()])

        # Display current price
        st.success(f"**{ticker}** | Spot: `${spot:.2f}` | Source: `{src}`")
        
        st.markdown("### 🏛️ Quant Stack Analysis")
        
        try:
            qs = QuantStackEngine(ticker, hist)
            
            # Layer 1: Macro Regime
            with st.spinner("Analyzing Macro Regime..."):
                regime, macro_score, macro_df, horsemen = qs.run_layer1_regime()
            
            # Layer 2: Kalman Filter (MUST be before strategy)
            kalman_df = qs.run_layer2_kalman()
            if kalman_df is not None and not kalman_df.empty:
                kalman_z = kalman_df['Kalman_Z'].iloc[-1]
            else:
                kalman_z = 0
            
            # Layer 3: GARCH Volatility
            sizing, vol_ratio, garch_status = qs.run_layer3_garch()
            
            # Layer 4: Jump Diffusion Tail Risk
            tail_risk, tail_score, jump_prob = qs.run_layer4_jump_diffusion()
            
            # ==========================================
            # RYAN MODEL (Volatility Squeeze Analysis)
            # ==========================================
            
            # Calculate Ryan Model metrics
            close = hist['Close']
            volume = hist['Volume'] if 'Volume' in hist.columns else pd.Series([0] * len(hist))
            
            # 1. Squeeze Detection (Bollinger Bandwidth)
            window = 20
            sma = close.rolling(window).mean()
            std = close.rolling(window).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            bandwidth = (upper_band - lower_band) / sma
            current_bw = bandwidth.iloc[-1] if len(bandwidth) > 0 else 0.2
            squeeze_threshold = 0.20
            is_squeeze = current_bw < squeeze_threshold
            squeeze_status = "COILED" if is_squeeze else "LOOSE"
            squeeze_score = max(0, (squeeze_threshold - current_bw) / squeeze_threshold * 100) if is_squeeze else 0
            
            # 2. Trend & Momentum
            curr_price = close.iloc[-1]
            trend = "BULLISH" if curr_price > sma.iloc[-1] else "BEARISH"
            ema_9 = close.ewm(span=9, adjust=False).mean()
            ema_21 = close.ewm(span=21, adjust=False).mean()
            momentum = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"
            
            # 3. Volume Velocity
            if len(volume) > 3:
                v3 = volume.rolling(3).mean().iloc[-1]
                v30 = volume.rolling(30).mean().iloc[-1] if len(volume) > 30 else v3
                v_status = "Speeding Up" if v3 > v30 else "Slowing Down"
                v_score = min(100, (v3 / v30 * 100)) if v30 > 0 else 50
            else:
                v_status = "Unknown"
                v_score = 50
            
            # 4. Relative Strength vs SPY
            try:
                spy_data = yf.download("SPY", period="60d", progress=False)['Close']
                ticker_60d_return = (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) > 60 else 0
                spy_60d_return = (spy_data.iloc[-1] / spy_data.iloc[-60] - 1) * 100 if len(spy_data) > 60 else 0
                rs_rating = ticker_60d_return - spy_60d_return
                rs_status = "Outperforming" if rs_rating > 0 else "Underperforming"
            except:
                rs_rating = 0
                rs_status = "N/A"
            
            # 5. Confidence Score (0-100)
            confidence = 0
            # Trend alignment (35 points)
            if trend == momentum:
                confidence += 35
            else:
                confidence += 15
            # Squeeze depth (25 points)
            if is_squeeze:
                confidence += 25
            # Volume velocity (20 points)
            if v_status == "Speeding Up":
                confidence += 20
            # Relative strength (20 points)
            if rs_rating > 0:
                confidence += 20
            
            confidence = min(100, max(0, confidence))
            
            st.markdown("---")
            
            # ==========================================
            # STRATEGY COMMAND CENTER (Top Priority)
            # ==========================================
            st.markdown("### 🎯 STRATEGY: Entry, Exit & Risk Management")
            
            # Calculate strategy based on all layers
            current_price = spot
            kalman_true_value = kalman_df['Kalman'].iloc[-1] if kalman_df is not None and not kalman_df.empty else spot
            
            # Entry logic
            should_enter = False
            entry_rationale = []
            risk_level = "HIGH"
            
            if regime == "BULL":
                entry_rationale.append("✅ Bullish macro environment (Four Horsemen)")
                if kalman_z < 2.0:
                    entry_rationale.append("✅ Fair valuation (Kalman Z < 2.0)")
                    if tail_risk in ["LOW", "MEDIUM"]:
                        entry_rationale.append(f"✅ Acceptable tail risk ({tail_risk})")
                        should_enter = True
                        risk_level = "LOW" if tail_risk == "LOW" else "MEDIUM"
                    else:
                        entry_rationale.append(f"⚠️ High tail risk ({tail_risk})")
                else:
                    entry_rationale.append("⛔ Overextended (Kalman Z > 2.0)")
            elif regime == "BEAR":
                entry_rationale.append("🛑 Bearish macro - WAIT for regime shift")
            else:
                entry_rationale.append("⚖️ Neutral macro - Selective entry only")
            
            # Calculate levels
            if should_enter:
                # Entry: Current price if fair, or wait for Kalman fair value
                entry_price = current_price if kalman_z < 1.5 else kalman_true_value
                
                # Stop loss: 2 ATR or 5% below entry, whichever is tighter
                if 'High' in kalman_df.columns and 'Low' in kalman_df.columns:
                    recent_high = kalman_df['High'].tail(20).max()
                    recent_low = kalman_df['Low'].tail(20).min()
                    atr_proxy = (recent_high - recent_low) / 20
                    stop_distance = min(atr_proxy * 2, entry_price * 0.05)
                else:
                    stop_distance = entry_price * 0.05
                
                stop_loss = entry_price - stop_distance
                
                # Take profit: Reasonable targets based on ATR
                if 'High' in kalman_df.columns and 'Low' in kalman_df.columns:
                    recent_high = kalman_df['High'].tail(20).max()
                    recent_low = kalman_df['Low'].tail(20).min()
                    atr = (recent_high - recent_low) / 20
                    target_1 = entry_price + (atr * 2)  # 2 ATR profit target
                    target_2 = entry_price + (atr * 3)  # 3 ATR stretch target
                else:
                    # Fallback: 5% and 10% profit targets
                    target_1 = entry_price * 1.05
                    target_2 = entry_price * 1.10
                
                # Position sizing from GARCH
                position_size_pct = min(100, cap * 0.01 * sizing * kelly)  # Max 1% risk, adjusted by GARCH
                shares = int(position_size_pct / entry_price)
                risk_level = tail_risk
                
                strat_color = "#10B981"
                strat_decision = "✅ ENTER LONG"
            else:
                entry_price = "WAIT"
                stop_loss = "N/A"
                target_1, target_2 = "N/A", "N/A"
                shares = 0
                position_size_pct = 0
                risk_level = tail_risk
                strat_color = "#EF4444"
                strat_decision = "🛑 DO NOT ENTER"
            
            # Display Strategy Box
            scol1, scol2 = st.columns([2, 1])
            
            with scol1:
                st.markdown(f"""
                <div class="signal-box" style="border-left-color: {strat_color}; background: {'#ECFDF5' if should_enter else '#FEF2F2'};">
                    <h2 style="margin:0; color:{strat_color};">{strat_decision}</h2>
                    <p style="margin:10px 0 0 0; line-height:1.6;">
                        {'<br/>'.join(entry_rationale)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with scol2:
                st.markdown(f"""
                <div style="border:1px solid #E2E8F0; padding:15px; border-radius:8px; background:#F8FAFC;">
                    <div style="margin-bottom:10px;">
                        <strong>Entry:</strong> ${entry_price if isinstance(entry_price, str) else f'{entry_price:.2f}'}<br/>
                        <strong>Stop Loss:</strong> ${stop_loss if isinstance(stop_loss, str) else f'{stop_loss:.2f}'}<br/>
                        <strong>Target 1:</strong> ${target_1 if isinstance(target_1, str) else f'{target_1:.2f}'}<br/>
                        <strong>Target 2:</strong> ${target_2 if isinstance(target_2, str) else f'{target_2:.2f}'}
                    </div>
                    <div style="border-top:1px solid #E2E8F0; padding-top:10px;">
                        <strong>Position Size:</strong> {shares} shares (${position_size_pct:.0f})<br/>
                        <strong>GARCH Sizing:</strong> {sizing:.1f}x<br/>
                        <strong>Risk Level:</strong> {risk_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- FOUR HORSEMEN DISPLAY (The Black Box) ---
            st.markdown("#### 🐴 The Four Horsemen (Institutional Flow Indicators)")
            st.caption("The invisible ratios that drive smart money - we don't care about price, we care about the environment.")
            
            h1, h2, h3, h4 = st.columns(4)
            
            horsemen_config = {
                'Credit': {
                    'label': 'Credit',
                    'subtitle': 'JNK / LQD',
                    'tooltip': 'Risk ON (rising) vs Risk OFF (falling)',
                    'col': h1
                },
                'Consumer': {
                    'label': 'Consumer',
                    'subtitle': 'XLY / XLP',
                    'tooltip': 'iPads (Bullish) vs Toothpaste (Bearish)',
                    'col': h2
                },
                'Growth': {
                    'label': 'Rate/Growth',
                    'subtitle': 'QQQ / TLT',
                    'tooltip': 'Tech (Growth) vs Bonds (Duration)',
                    'col': h3
                },
                'Fear': {
                    'label': 'Inflation',
                    'subtitle': 'SPY / GLD',
                    'tooltip': 'Stocks (Risk) vs Gold (Safety)',
                    'col': h4
                }
            }
            
            for horse_name, config in horsemen_config.items():
                with config['col']:
                    if horse_name in horsemen and horsemen[horse_name]:
                        z = horsemen[horse_name]['z']
                        val = horsemen[horse_name]['value']
                        
                        # Color based on Z-score (positive = bullish, negative = bearish)
                        if z > 1.0:
                            sentiment, color = "BULLISH", "#10B981"
                        elif z < -1.0:
                            sentiment, color = "BEARISH", "#EF4444"
                        else:
                            sentiment, color = "NEUTRAL", "#F59E0B"
                        
                        st.markdown(f"""
                        <div style="border:1px solid #E2E8F0; padding:10px; border-radius:6px; background:linear-gradient(135deg, {color}10, {color}20);">
                            <div style="font-size:0.75rem; color:#64748B; margin-bottom:3px;">{config['label']}</div>
                            <div style="font-size:1.1rem; font-weight:700; color:{color};">{sentiment}</div>
                            <div style="font-size:0.7rem; color:#94A3B8; margin-top:2px;">
                                {config['subtitle']}<br/>Z: {z:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("N/A")
            
            st.markdown("---")
            
            # --- RYAN MODEL METRICS ---
            st.markdown("#### 📊 Ryan Model Output")
            st.caption("Volatility Squeeze + Momentum + Volume Velocity + Relative Strength")
            
            rm1, rm2, rm3, rm4, rm5, rm6 = st.columns(6)
            
            with rm1:
                st.metric("Squeeze", squeeze_status, 
                         delta=f"{current_bw:.3f}" if is_squeeze else "Normal",
                         delta_color="normal" if is_squeeze else "off")
            
            with rm2:
                trend_color = "normal" if trend == "BULLISH" else "inverse"
                st.metric("Trend", trend, delta_color=trend_color)
            
            with rm3:
                mom_color = "normal" if momentum == "BULLISH" else "inverse"
                st.metric("Momentum", momentum, delta_color=mom_color)
            
            with rm4:
                vol_color = "normal" if v_status == "Speeding Up" else "inverse"
                st.metric("Volume", v_status, delta_color=vol_color)
            
            with rm5:
                rs_color = "normal" if rs_rating > 0 else "inverse"
                st.metric("RS Rating", f"{rs_rating:.1f}%", delta=rs_status, delta_color=rs_color)
            
            with rm6:
                conf_color = "normal" if confidence >= 70 else "inverse" if confidence < 50 else "off"
                st.metric("Confidence", f"{confidence}%", delta_color=conf_color)
            
            st.markdown("---")
            
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
        st.markdown("### 📊 Market Indices")
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
        
        # --- FOUR HORSEMEN INDICATORS ---
        st.markdown("---")
        st.markdown("#### 🐴 Four Horsemen (Institutional Flows)")
        
        # Calculate horsemen using QuantStackEngine
        try:
            temp_qs = QuantStackEngine("SPY", pd.DataFrame())  # Dummy ticker, we only need macro
            _, _, _, horsemen_dash = temp_qs.run_layer1_regime()
            
            if horsemen_dash:
                hcol1, hcol2, hcol3, hcol4 = st.columns(4)
                
                horsemen_labels = {
                    'Credit': ('🏛️ Credit Risk', 'JNK/LQD', hcol1),
                    'Consumer': ('🛍️ Consumer Spending', 'XLY/XLP', hcol2),
                    'Growth': ('📈 Rate/Growth', 'QQQ/TLT', hcol3),
                    'Fear': ('🪙 Inflation Fear', 'SPY/GLD', hcol4)
                }
                
                for horse_name, (label, ratio, col) in horsemen_labels.items():
                    with col:
                        if horse_name in horsemen_dash:
                            z = horsemen_dash[horse_name]['z']
                            val = horsemen_dash[horse_name]['value']
                            
                            if z > 1.0:
                                sentiment, delta_color = "🟢 Bullish", "normal"
                            elif z < -1.0:
                                sentiment, delta_color = "🔴 Bearish", "inverse"
                            else:
                                sentiment, delta_color = "🟡 Neutral", "off"
                            
                            st.metric(
                                label=f"{label}",
                                value=f"{val:.3f}",
                                delta=sentiment,
                                delta_color=delta_color,
                                help=f"{ratio} | Z-Score: {z:.2f}"
                            )
        except Exception:
            st.info("Loading institutional flow data...")

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
                <div style="background: #1E293B; 
                            padding: 24px; border-radius: 16px; border: 1px solid #334155; text-align: center;
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">{fear_emoji}</div>
                    <h2 style="margin: 0; color: {fear_color}; font-size: 1.8rem; letter-spacing: 2px;">{fear_level}</h2>
                    <p style="margin: 12px 0 0 0; color: #94A3B8; font-size: 0.95rem; font-weight: 500;">
                        Score: <span style="color: #FFFFFF; font-weight: 700;">{composite_score:.0f}/100</span> · VIX: <span style="color: #FFFFFF; font-weight: 700;">{vix_level:.1f}</span><br/>
                        <span style="color: #60A5FA;">{bullish_count}/{total_indices} indices up</span>
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
    # MODULE D: THE HUNTER (Wide Net Scanner)
    # ==========================================
    elif mode == "🦅 Hunter":
        st.title("🦅 The Hunter: Market Scanner")
        st.caption("Cast the net wide | Momentum + Squeeze + Trap Detection")

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
            "NFLX", "BKNG", "ABNB", "SHOP", "SPOT", "SNAP", "PINS", "RBLX", "U", "TTD", "TEAM", "DOCU", "ZM",
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

        if st.button("🚀 Initialize Scan Sequence"):
            progress_text = "Connecting to Neural Lattice... Please wait."
            my_bar = st.progress(0, text=progress_text)

            with st.spinner(f"Scanning {len(TICKER_LIST)} Assets..."):
                my_bar.progress(30, text="Downloading Batch Data...")
                df_results = batch_process_tickers(TICKER_LIST, scan_window, scan_sqz_thresh)
                my_bar.progress(90, text="Processing Results...")

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

                    def style_longs(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for idx in df.index:
                            conf = df.loc[idx, "Confidence"]
                            action = df.loc[idx, "Action"]
                            if "BUY" in str(action):
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.4); font-weight: bold;"
                            elif conf >= 90:
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.3);"
                            elif conf >= 85:
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.2);"
                            elif conf >= 80:
                                styles.loc[idx, :] = "background-color: rgba(16, 185, 129, 0.1);"
                        return styles

                    st.subheader(f"🟢 Long Setups ({len(longs)})")
                    if not longs.empty:
                        longs_sorted = longs.sort_values(by="Confidence", ascending=False)
                        st.dataframe(
                            longs_sorted.style.apply(style_longs, axis=None).format({"Price": "${:.2f}", "Bandwidth": "{:.4f}", "RS_vs_SPY": "{:.2%}"}),
                            use_container_width=True,
                            column_config={
                                "Confidence": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100),
                            },
                            column_order=("Ticker", "Action", "Confidence", "Health", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                            hide_index=True,
                        )
                        top_buys = longs_sorted[longs_sorted["Action"].str.contains("BUY")]
                        if not top_buys.empty:
                            st.success(f"🎯 **TOP PICKS:** {', '.join(top_buys['Ticker'].head(5).tolist())}")
                    else:
                        st.info(f"No Long setups found (Conf > {min_confidence}%).")

                    st.markdown("---")

                    def style_shorts(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for idx in df.index:
                            conf = df.loc[idx, "Confidence"]
                            action = df.loc[idx, "Action"]
                            if "SHORT" in str(action):
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.4); font-weight: bold;"
                            elif conf >= 90:
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.3);"
                            elif conf >= 85:
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.2);"
                            elif conf >= 80:
                                styles.loc[idx, :] = "background-color: rgba(239, 68, 68, 0.1);"
                        return styles

                    st.subheader(f"🔴 Short Setups ({len(shorts)})")
                    if not shorts.empty:
                        shorts_sorted = shorts.sort_values(by="Confidence", ascending=False)
                        st.dataframe(
                            shorts_sorted.style.apply(style_shorts, axis=None).format({"Price": "${:.2f}", "Bandwidth": "{:.4f}", "RS_vs_SPY": "{:.2%}"}),
                            use_container_width=True,
                            column_config={"Confidence": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100)},
                            column_order=("Ticker", "Action", "Confidence", "Health", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                            hide_index=True,
                        )
                        top_shorts = shorts_sorted[shorts_sorted["Action"].str.contains("SHORT")]
                        if not top_shorts.empty:
                            st.error(f"🎯 **TOP SHORTS:** {', '.join(top_shorts['Ticker'].head(5).tolist())}")
                    else:
                        st.info(f"No Short setups found (Conf > {min_confidence}%).")

                    with st.expander("📂 View Full Scan Results (All Assets)"):
                        st.dataframe(
                            df_results.sort_values(by="Confidence", ascending=False).style.format({"Price": "${:.2f}", "Confidence": "{:.0f}%"}),
                            use_container_width=True,
                        )
                else:
                    st.error("Scan returned no data. Check API connection.")
        else:
            st.info("🦅 **Ready to Hunt.** Click to scan 100+ assets using momentum, squeeze, and trap detection.")


    # ==========================================
    # MODULE E: RYAN MODEL 2.0 (Quant Intelligence Funnel)
    # ==========================================
    elif mode == "🧠 Ryan Model 2.0":
        st.title("🧠 Ryan Model 2.0: Quantitative Intelligence Funnel")
        st.caption("Five Composite Quant Factors | Value + Momentum + Low Vol + PEAD + RSI")
        
        st.markdown("""
        ### 📋 The Philosophical Objective
        
        The Ryan Model 2.0 calculates **5 Composite Quant Factors** on a curated stock universe:
        1. **Valuation + Sentiment** (P/E based)
        2. **Post-Earnings Drift** (PEAD signal)
        3. **Low Volatility Anomaly** (Beta < 1.0)
        4. **RSI-2 Mean Reversion** (Oversold in uptrend)
        5. **Value + Momentum** (Magic Formula proxy)
        
        *Each factor scores 0-100, weighted into a final Conviction Score.*
        """)

        # ADJUSTABLE FILTERS
        with st.expander("🛠️ Screener Configuration", expanded=True):
            st.markdown("**Filter Settings**")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                min_price = st.slider("Min Price ($)", 1, 50, 5, key="ryan_min_price")
                max_price = st.slider("Max Price ($)", 50, 2000, 500, key="ryan_max_price")
            with fc2:
                min_conviction = st.slider("Min Conviction Score", 20, 90, 40, key="ryan_conv")
                max_results = st.slider("Max Results", 10, 100, 30, key="ryan_max")
            with fc3:
                universe_choice = st.selectbox("Stock Universe", [
                    "🌟 Quality 100 (Recommended)",
                    "📈 Mega Cap Tech",
                    "💰 Financials",
                    "🏥 Healthcare",
                    "⚡ High Beta / Momentum"
                ], key="ryan_universe")
        
        # STOCK UNIVERSES (No Finviz dependency)
        STOCK_UNIVERSES = {
            "🌟 Quality 100 (Recommended)": [
                # Mega-cap Tech
                "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
                "ORCL", "ADBE", "CRM", "INTC", "QCOM", "TXN", "MU", "NOW", "PANW", "SNOW",
                # Financials
                "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "AXP", "BLK", "C",
                # Healthcare
                "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "AMGN", "TMO", "DHR", "ABT",
                # Consumer
                "WMT", "COST", "PG", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX",
                # Industrial / Energy
                "CAT", "DE", "HON", "GE", "BA", "XOM", "CVX", "COP", "SLB", "LMT",
                # Growth / Momentum
                "PLTR", "COIN", "MSTR", "DKNG", "ROKU", "SQ", "SHOP", "SPOT", "ABNB", "UBER"
            ],
            "📈 Mega Cap Tech": [
                "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "ADBE",
                "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "NOW", "PANW", "SNOW", "NET"
            ],
            "💰 Financials": [
                "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "AXP", "BLK", "C",
                "PYPL", "SCHW", "CME", "ICE", "SPGI", "MCO", "COF", "USB", "PNC", "TFC"
            ],
            "🏥 Healthcare": [
                "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "AMGN", "TMO", "DHR", "ABT",
                "BMY", "GILD", "ISRG", "VRTX", "REGN", "MDT", "SYK", "BSX", "ELV", "CI"
            ],
            "⚡ High Beta / Momentum": [
                "TSLA", "NVDA", "AMD", "MSTR", "COIN", "PLTR", "DKNG", "ROKU", "SQ", "SHOP",
                "AFRM", "UPST", "CVNA", "MARA", "RIOT", "CLSK", "HOOD", "SOFI", "AI", "IONQ"
            ]
        }
        
        def get_stock_universe(choice):
            """Get stock list based on user selection"""
            return STOCK_UNIVERSES.get(choice, STOCK_UNIVERSES["🌟 Quality 100 (Recommended)"])

        @st.cache_data(ttl=1800)
        def calculate_quant_factors(ticker, spy_data):
            """Stage 2: Calculate 5 Composite Quant Factors"""
            try:
                # Fetch ticker data
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if hist.empty or len(hist) < 60:
                    return None
                    
                info = stock.info
                current_price = hist['Close'].iloc[-1]
                
                # Initialize factor scores
                factors = {
                    'ticker': ticker,
                    'price': current_price,
                    'pe_putcall': 0,
                    'pead': 0,
                    'low_vol': 0,
                    'rsi2': 0,
                    'value_momentum': 0,
                    'conviction': 0
                }
                
                # --- FACTOR 1: Forward P/E / Put-Call Ratio (Valuation + Sentiment) ---
                try:
                    fwd_pe = info.get('forwardPE', None)
                    # Note: Put/Call ratio not directly available in yfinance
                    # We'll use a proxy: implied volatility sentiment
                    if fwd_pe and fwd_pe > 0 and fwd_pe < 50:
                        # Lower P/E = higher score
                        factors['pe_putcall'] = min(100, (30 / fwd_pe) * 100)
                except:
                    factors['pe_putcall'] = 50  # Neutral
                
                # --- FACTOR 2: Post-Earnings Announcement Drift (PEAD) ---
                try:
                    # Check if earnings date was recent (within 40 days)
                    earnings_dates = stock.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        latest_earnings = earnings_dates.index[0]
                        days_since = (pd.Timestamp.now() - latest_earnings).days
                        
                        # Check if price is above 50-day SMA (momentum)
                        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                        if 0 < days_since <= 40 and current_price > sma_50:
                            factors['pead'] = 100  # Strong PEAD
                        elif current_price > sma_50:
                            factors['pead'] = 50  # Momentum but no recent earnings
                        else:
                            factors['pead'] = 30  # Below SMA
                    else:
                        factors['pead'] = 40  # No earnings data baseline
                except:
                    factors['pead'] = 40  # Neutral baseline
                
                # --- FACTOR 3: Low Volatility Anomaly ---
                try:
                    # Calculate Beta
                    spy_aligned = spy_data['Close'].reindex(hist.index).ffill().dropna()
                    stock_aligned = hist['Close'].reindex(spy_aligned.index).dropna()
                    spy_aligned = spy_aligned.loc[stock_aligned.index]
                    
                    if len(stock_aligned) > 60:
                        rets_stock = stock_aligned.pct_change().dropna()
                        rets_spy = spy_aligned.pct_change().dropna()
                        common = rets_stock.index.intersection(rets_spy.index)
                        
                        if len(common) > 30:
                            cov = np.cov(rets_stock.loc[common], rets_spy.loc[common])[0][1]
                            var = np.var(rets_spy.loc[common])
                            beta = cov / var
                            
                            # Low beta = high score, but give baseline for all
                            if beta < 0.7:
                                factors['low_vol'] = 100
                            elif beta < 1.0:
                                factors['low_vol'] = 70
                            elif beta < 1.3:
                                factors['low_vol'] = 50
                            else:
                                factors['low_vol'] = 30  # Baseline for high beta
                except:
                    factors['low_vol'] = 40  # Neutral baseline
                
                # --- FACTOR 4: RSI-2 Mean Reversion ---
                try:
                    # Calculate RSI-2
                    closes = hist['Close']
                    delta = closes.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
                    rs = gain / loss
                    rsi_2 = 100 - (100 / (1 + rs))
                    
                    # Check 200 SMA (long-term trend)
                    sma_200 = closes.rolling(200).mean().iloc[-1]
                    
                    if current_price > sma_200 and rsi_2.iloc[-1] < 10:
                        factors['rsi2'] = 100  # Extreme oversold in uptrend
                    elif current_price > sma_200 and rsi_2.iloc[-1] < 20:
                        factors['rsi2'] = 80  # Moderately oversold
                    elif current_price > sma_200 and rsi_2.iloc[-1] < 40:
                        factors['rsi2'] = 60  # Slightly oversold in uptrend
                    elif current_price > sma_200:
                        factors['rsi2'] = 40  # Uptrend but not oversold
                    else:
                        factors['rsi2'] = 20  # Downtrend baseline
                except:
                    factors['rsi2'] = 30  # Neutral baseline
                
                # --- FACTOR 5: Value + Momentum (Magic Formula) ---
                try:
                    # Value: EV/EBITDA proxy (use P/E as fallback)
                    pe = info.get('trailingPE', info.get('forwardPE', None))
                    
                    # Momentum: 12-month return
                    if len(hist) >= 252:
                        ret_12m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1)
                    else:
                        ret_12m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                    
                    # Combined score - give baseline for all stocks
                    value_score = 20  # Baseline
                    if pe and pe > 0:
                        if pe < 12: value_score = 60
                        elif pe < 18: value_score = 50
                        elif pe < 25: value_score = 40
                        elif pe < 35: value_score = 30
                    
                    momentum_score = 10  # Baseline
                    if ret_12m > 0.30: momentum_score = 50
                    elif ret_12m > 0.15: momentum_score = 40
                    elif ret_12m > 0.05: momentum_score = 30
                    elif ret_12m > 0: momentum_score = 25
                    
                    factors['value_momentum'] = value_score + momentum_score
                except:
                    factors['value_momentum'] = 40  # Neutral baseline
                
                # --- COMPOSITE CONVICTION SCORE ---
                # Weighted: 25% PE/PC, 15% PEAD, 15% LowVol, 10% RSI2, 35% Value+Momentum
                conviction = (
                    factors['pe_putcall'] * 0.25 +
                    factors['pead'] * 0.15 +
                    factors['low_vol'] * 0.15 +
                    factors['rsi2'] * 0.10 +
                    factors['value_momentum'] * 0.35
                )
                
                factors['conviction'] = int(conviction)
                
                return factors
                
            except Exception as e:
                return None

        def run_quant_pipeline(universe, min_p, max_p):
            """Calculate quant factors for selected stock universe"""
            import random
            tickers = list(universe)  # Make a copy
            random.shuffle(tickers)   # Randomize order so not biased toward A-Z
            
            if not tickers:
                return pd.DataFrame(), "No tickers in selected universe"
            
            # Fetch SPY for beta calculations
            spy_data = yf.download("SPY", period="1y", progress=False)
            
            # Calculate Quant Factors for ALL stocks in universe
            results = []
            for ticker in tickers:
                try:
                    factors = calculate_quant_factors(ticker, spy_data)
                    if factors:
                        # Apply price filter
                        if min_p <= factors['price'] <= max_p:
                            results.append(factors)
                except:
                    continue
            
            df = pd.DataFrame(results)
            return df, None

        if st.button("🚀 Run Quant Analysis", type="primary", use_container_width=True, key="ryan_scan"):
            selected_universe = get_stock_universe(universe_choice)
            
            progress_bar = st.progress(0, text="Initializing...")
            progress_bar.progress(10, text=f"Analyzing {len(selected_universe)} stocks...")
            
            with st.spinner(f"Calculating 5 Quant Factors for {len(selected_universe)} stocks..."):
                df_results, error = run_quant_pipeline(selected_universe, min_price, max_price)
                progress_bar.progress(100, text="Analysis Complete!")
                progress_bar.empty()
                
                if error:
                    st.error(error)
                elif df_results is not None and not df_results.empty:
                    df_results = df_results.sort_values(by='conviction', ascending=False)
                    
                    # Filter by min conviction
                    df_filtered = df_results[df_results['conviction'] >= min_conviction]
                    
                    st.markdown(f"### 📊 Quant Factor Analysis ({len(df_filtered)} High-Conviction Targets)")
                    
                    st.dataframe(
                        df_filtered.head(max_results).style.format({
                            'price': '${:.2f}',
                            'pe_putcall': '{:.0f}',
                            'pead': '{:.0f}',
                            'low_vol': '{:.0f}',
                            'rsi2': '{:.0f}',
                            'value_momentum': '{:.0f}',
                            'conviction': '{:.0f}'
                        }),
                        use_container_width=True,
                        column_config={
                            'ticker': st.column_config.TextColumn("Ticker", help="Symbol"),
                            'price': st.column_config.NumberColumn("Price", format="$%.2f"),
                            'conviction': st.column_config.ProgressColumn("Conviction", min_value=0, max_value=100, format="%d"),
                            'pe_putcall': st.column_config.NumberColumn("P/E Factor", help="Valuation + Sentiment"),
                            'pead': st.column_config.NumberColumn("PEAD", help="Post-Earnings Drift"),
                            'low_vol': st.column_config.NumberColumn("Low Vol", help="Low Volatility Anomaly"),
                            'rsi2': st.column_config.NumberColumn("RSI-2", help="Mean Reversion"),
                            'value_momentum': st.column_config.NumberColumn("Val+Mom", help="Value + Momentum (Magic Formula)"),
                        },
                        hide_index=True
                    )
                    
                    if not df_filtered.empty:
                        top_picks = df_filtered.head(5)['ticker'].tolist()
                        st.success(f"🎯 **TOP CONVICTION TARGETS:** {', '.join(top_picks)}")
                else:
                    st.info("No results from screening pipeline.")
        else:
            st.info("🧠 **Ready to Deploy.** Click to initialize the Two-Pronged Quant Intelligence Funnel.")
            
            st.markdown("""
            ---
            ### 🎯 The Five Composite Quant Factors
            
            **1. Valuation + Sentiment (P/E / Put-Call)**  
            - **Logic**: Low Forward P/E (cheap) combined with bearish sentiment (fear) = opportunity
            - **Thesis**: "Cheap + Hated = Mean Reversion Setup"
            
            **2. Post-Earnings Announcement Drift (PEAD)**  
            - **Logic**: Stocks that beat earnings by >10% continue drifting upward for 30-60 days
            - **Thesis**: "Markets are inefficient at pricing news instantly"
            
            **3. Low Volatility Anomaly**  
            - **Logic**: Low-beta stocks (β < 0.7) produce better risk-adjusted returns
            - **Thesis**: "Boring stocks outperform because fund managers chase high-beta names"
            
            **4. RSI-2 Mean Reversion**  
            - **Logic**: Extreme short-term oversold (RSI-2 < 10) in long-term uptrend (>200 SMA)
            - **Thesis**: "Algorithms overreact in the short term, creating bounce opportunities"
            
            **5. Value + Momentum (Magic Formula)**  
            - **Logic**: Cheap stocks (low EV/EBITDA) with momentum (positive 12-month return)
            - **Thesis**: "Filters out value traps (cheap but dying) and bubble stocks (expensive momentum)"
            
            ---
            
            ### 📈 Composite Scoring
            
            **Weighted Conviction Score (0-100%)**:
            - 25% P/E / Put-Call (Valuation + Sentiment)
            - 15% PEAD (Post-Earnings Drift)
            - 15% Low Vol Anomaly
            - 10% RSI-2 (Mean Reversion)
            - 35% Value + Momentum (Magic Formula)
            
            ---
            
            ### 💡 System Architecture
            
            *"The Ryan Model 2.0 isn't just a trading tool; it's a **Quantitative Intelligence Funnel**. 
            By combining Finviz efficiency screening with composite mathematical factors, I've built a system 
            that transforms qualitative 'chart patterns' into quantitative 'statistical probabilities.' 
            This bridges traditional technical analysis with institutional-grade data science."*
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
    # MODULE F: PAIRS (Correlation Analysis)
    # ==========================================
    elif mode == "🔗 Pairs":
        st.title("🔗 Pairs: Correlation Analysis")
        st.caption("Rolling 3-Month Correlation Heatmap | Four Horsemen + Hunter Universe")
        
        st.markdown("""
        ### 📋 Pairs Trading & Correlation
        
        This module calculates **rolling 3-month correlations** between:
        - **Four Horsemen ETFs**: JNK, LQD, XLY, XLP, QQQ, TLT, SPY, GLD (regime indicators)
        - **Hunter Stock Universe**: 60+ liquid stocks across sectors
        
        High correlation pairs move together. Low/negative correlation pairs diverge.
        """)
        
        # Configuration
        with st.expander("🛠️ Configuration", expanded=True):
            pc1, pc2 = st.columns(2)
            with pc1:
                lookback_months = st.slider("Lookback Period (Months)", 1, 12, 3, key="pairs_lookback")
                min_corr = st.slider("Min Correlation Filter", -1.0, 0.5, -1.0, 0.1, key="pairs_min_corr")
            with pc2:
                universe_type = st.selectbox("Asset Universe", [
                    "🏛️ Four Horsemen ETFs Only (8)",
                    "🦅 Hunter Stocks (from Scanner)",
                    "🌐 Combined Universe"
                ], key="pairs_universe")
        
        # Use the same universes as the rest of the app
        FOUR_HORSEMEN = ['JNK', 'LQD', 'XLY', 'XLP', 'QQQ', 'TLT', 'SPY', 'GLD']
        
        # Reference the ScannerEngine WATCHLIST so it stays in sync
        HUNTER_UNIVERSE = ScannerEngine.WATCHLIST
        
        def get_pairs_universe(choice):
            if "Four Horsemen" in choice:
                return FOUR_HORSEMEN
            elif "Hunter" in choice:
                return HUNTER_UNIVERSE[:50]  # Limit for performance
            else:
                return FOUR_HORSEMEN + HUNTER_UNIVERSE[:40]
        
        if st.button("📊 Generate Correlation Heatmap", type="primary", use_container_width=True, key="pairs_scan"):
            selected_assets = get_pairs_universe(universe_type)
            
            # Calculate date range (rolling 3 months)
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(months=lookback_months)
            
            with st.spinner(f"Fetching {len(selected_assets)} assets for {lookback_months}-month period..."):
                try:
                    # Download price data
                    prices = yf.download(
                        selected_assets,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False
                    )["Close"]
                    
                    if prices.empty:
                        st.error("No price data returned. Try again.")
                    else:
                        # Calculate returns and correlation
                        returns = prices.pct_change().dropna()
                        corr = returns.corr()
                        
                        # Display stats
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Assets Analyzed", len(corr.columns))
                        with c2:
                            st.metric("Trading Days", len(returns))
                        with c3:
                            avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
                            st.metric("Avg Correlation", f"{avg_corr:.2f}")
                        
                        # Create Plotly heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdYlGn',
                            zmid=0,
                            text=np.round(corr.values, 2),
                            texttemplate="%{text}",
                            textfont={"size": 8},
                            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>"
                        ))
                        
                        fig.update_layout(
                            title=f"Rolling {lookback_months}-Month Correlation Matrix",
                            height=max(500, len(corr.columns) * 20),
                            xaxis_title="",
                            yaxis_title="",
                            xaxis={'side': 'bottom'},
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find extreme pairs
                        st.markdown("---")
                        st.subheader("🔍 Notable Pairs")
                        
                        # Get upper triangle pairs
                        pairs_list = []
                        for i in range(len(corr.columns)):
                            for j in range(i+1, len(corr.columns)):
                                pairs_list.append({
                                    'Pair': f"{corr.columns[i]} / {corr.columns[j]}",
                                    'Asset1': corr.columns[i],
                                    'Asset2': corr.columns[j],
                                    'Correlation': corr.iloc[i, j]
                                })
                        
                        pairs_df = pd.DataFrame(pairs_list)
                        pairs_df = pairs_df[pairs_df['Correlation'] >= min_corr]
                        
                        tab1, tab2 = st.tabs(["📈 Highest Correlation", "📉 Lowest Correlation"])
                        
                        with tab1:
                            high_corr = pairs_df.nlargest(15, 'Correlation')
                            st.caption("*These pairs move together - good for confirmation, bad for diversification*")
                            st.dataframe(
                                high_corr[['Pair', 'Correlation']],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Correlation': st.column_config.ProgressColumn(
                                        'Correlation', format="%.2f", min_value=-1, max_value=1
                                    )
                                }
                            )
                        
                        with tab2:
                            low_corr = pairs_df.nsmallest(15, 'Correlation')
                            st.caption("*These pairs diverge - potential spread trades or hedges*")
                            st.dataframe(
                                low_corr[['Pair', 'Correlation']],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Correlation': st.column_config.ProgressColumn(
                                        'Correlation', format="%.2f", min_value=-1, max_value=1
                                    )
                                }
                            )
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        else:
            st.info("🔗 **Click to analyze correlations** between assets over your selected time period.")
            
            st.markdown("""
            ### How To Use
            
            1. **Select Universe**: Choose which assets to analyze
            2. **Set Lookback**: Rolling period captures current regime
            3. **Generate Heatmap**: Click to run analysis
            
            **Interpretation:**
            - 🟢 **Green (+1.0)**: Highly correlated (move together)
            - 🟡 **Yellow (0.0)**: Uncorrelated (independent)
            - 🔴 **Red (-1.0)**: Negatively correlated (move opposite)
            
            **Use Cases:**
            - Find hedging pairs (negative correlation)
            - Identify regime shifts (changing correlations)
            - Spot crowded trades (high correlation clusters)
            """)
        
        # ==========================================
        # PAIRS DEEP DIVE SECTION
        # ==========================================
        st.markdown("---")
        st.subheader("🔬 Pairs Deep Dive Analysis")
        st.caption("Cointegration Test + Half-Life + Z-Score Signals + Backtest")
        
        with st.expander("⚙️ Deep Dive Configuration", expanded=True):
            dd1, dd2 = st.columns(2)
            with dd1:
                pair_a = st.text_input("Stock A (e.g. MA)", value="MA", key="pair_a").upper().strip()
                pair_b = st.text_input("Stock B (e.g. V)", value="V", key="pair_b").upper().strip()
            with dd2:
                dd_lookback = st.selectbox("Lookback Period", ["6 Months", "1 Year", "2 Years"], index=1, key="dd_lookback")
                entry_z = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1, key="entry_z")
                exit_z = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1, key="exit_z")
        
        if st.button("🔬 Run Pairs Deep Dive", type="primary", use_container_width=True, key="pairs_deep"):
            if not pair_a or not pair_b or pair_a == pair_b:
                st.error("Please enter two different stock symbols.")
            else:
                # Determine start date
                if "6 Months" in dd_lookback:
                    start = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
                elif "2 Years" in dd_lookback:
                    start = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
                else:
                    start = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
                
                with st.spinner(f"Analyzing {pair_a} vs {pair_b}..."):
                    try:
                        # Download prices
                        prices = yf.download([pair_a, pair_b], start=start, progress=False)["Close"].dropna()
                        
                        if prices.empty or len(prices) < 60:
                            st.error("Insufficient data. Try different symbols or longer lookback.")
                        else:
                            logp = np.log(prices)
                            rets = prices.pct_change().dropna()
                            
                            # ========== COINTEGRATION TEST ==========
                            st.markdown("---")
                            st.markdown("### 📊 Cointegration Test (Engle-Granger)")
                            
                            if STATSMODELS_AVAILABLE:
                                score, pvalue, crit = coint(logp[pair_a].dropna(), logp[pair_b].dropna())
                                
                                coint_c1, coint_c2, coint_c3 = st.columns(3)
                                with coint_c1:
                                    st.metric("Test Statistic", f"{score:.3f}")
                                with coint_c2:
                                    color = "normal" if pvalue < 0.05 else "off"
                                    st.metric("P-Value", f"{pvalue:.4f}", delta="✓ Cointegrated" if pvalue < 0.05 else "✗ Not Cointegrated", delta_color=color)
                                with coint_c3:
                                    st.metric("5% Critical Value", f"{crit[1]:.3f}")
                                
                                if pvalue < 0.05:
                                    st.success(f"✅ {pair_a} and {pair_b} are cointegrated (p={pvalue:.4f})")
                                else:
                                    st.warning(f"⚠️ {pair_a} and {pair_b} are NOT cointegrated (p={pvalue:.4f})")
                            else:
                                st.warning("statsmodels not installed. Install with: `pip install statsmodels`")
                                pvalue = 1.0
                            
                            # ========== HEDGE RATIO & SPREAD ==========
                            BETA_LOOKBACK = min(252, len(prices) - 1)
                            Z_LOOKBACK = min(126, len(prices) - 1)
                            CORR_LOOKBACK = 60
                            
                            cov_ab = logp[pair_a].rolling(BETA_LOOKBACK).cov(logp[pair_b])
                            var_b = logp[pair_b].rolling(BETA_LOOKBACK).var()
                            beta = (cov_ab / var_b).rename("beta")
                            
                            spread = (logp[pair_a] - beta * logp[pair_b]).rename("spread")
                            
                            # Z-score
                            mu = spread.rolling(Z_LOOKBACK).mean()
                            sig = spread.rolling(Z_LOOKBACK).std()
                            z = ((spread - mu) / sig).rename("zscore")
                            
                            # ========== HALF-LIFE ==========
                            st.markdown("---")
                            st.markdown("### ⏱️ Half-Life of Mean Reversion")
                            
                            spread_lag = spread.shift(1)
                            delta_spread = spread - spread_lag
                            hl_df = pd.concat([delta_spread.rename("dS"), spread_lag.rename("S_lag")], axis=1).dropna()
                            
                            x = hl_df["S_lag"].values
                            y = hl_df["dS"].values
                            x_mean = x.mean()
                            y_mean = y.mean()
                            b_slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
                            
                            if b_slope < 0:
                                half_life = -np.log(2) / b_slope
                            else:
                                half_life = np.nan
                            
                            hl_c1, hl_c2 = st.columns(2)
                            with hl_c1:
                                st.metric("Regression Slope", f"{b_slope:.5f}")
                            with hl_c2:
                                if np.isfinite(half_life):
                                    hl_status = "✓ Good" if half_life < 60 else "Slow"
                                    st.metric("Half-Life (days)", f"{half_life:.1f}", delta=hl_status)
                                else:
                                    st.metric("Half-Life", "N/A", delta="No mean reversion")
                            
                            # ========== ROLLING CORRELATION ==========
                            st.markdown("---")
                            st.markdown("### 📈 Rolling Correlation")
                            
                            roll_corr = rets[pair_a].rolling(CORR_LOOKBACK).corr(rets[pair_b]).rename("roll_corr")
                            
                            corr_fig = go.Figure()
                            corr_fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode='lines', name='Rolling Corr', line=dict(color='#3B82F6', width=2)))
                            corr_fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="0.80 threshold")
                            corr_fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
                            corr_fig.update_layout(height=300, title=f"{CORR_LOOKBACK}-Day Rolling Correlation", template="plotly_white", yaxis_range=[-1, 1])
                            st.plotly_chart(corr_fig, use_container_width=True)
                            
                            current_corr = roll_corr.iloc[-1] if len(roll_corr) > 0 else 0
                            st.metric("Current Rolling Correlation", f"{current_corr:.3f}")
                            
                            # ========== Z-SCORE CHART ==========
                            st.markdown("---")
                            st.markdown("### 📉 Spread Z-Score")
                            
                            z_clean = z.dropna()
                            z_fig = go.Figure()
                            z_fig.add_trace(go.Scatter(x=z_clean.index, y=z_clean.values, mode='lines', name='Z-Score', line=dict(color='#1E293B', width=1.5)))
                            z_fig.add_hline(y=entry_z, line_dash="dash", line_color="red", annotation_text=f"Entry +{entry_z}")
                            z_fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", annotation_text=f"Entry -{entry_z}")
                            z_fig.add_hline(y=exit_z, line_dash="dot", line_color="orange")
                            z_fig.add_hline(y=-exit_z, line_dash="dot", line_color="orange")
                            z_fig.add_hline(y=0, line_color="gray", line_width=0.5)
                            z_fig.update_layout(height=350, title="Spread Z-Score with Entry/Exit Bands", template="plotly_white")
                            st.plotly_chart(z_fig, use_container_width=True)
                            
                            # ========== BACKTEST ==========
                            st.markdown("---")
                            st.markdown("### 💰 Simple Backtest")
                            
                            # Build data frame
                            data = pd.concat([prices, rets.add_prefix("ret_"), beta, spread, z, roll_corr], axis=1).dropna()
                            data["entry_allowed"] = (data["roll_corr"] >= 0.80)
                            
                            STOP_Z = 3.5
                            pos = 0
                            positions = []
                            for zi, allowed in zip(data["zscore"].values, data["entry_allowed"].values):
                                if pos == 0:
                                    if allowed:
                                        if zi > entry_z:
                                            pos = -1
                                        elif zi < -entry_z:
                                            pos = 1
                                else:
                                    if abs(zi) < exit_z:
                                        pos = 0
                                    elif abs(zi) > STOP_Z:
                                        pos = 0
                                    elif not allowed:
                                        pos = 0
                                positions.append(pos)
                            
                            data["position"] = positions
                            data["pos_lag"] = data["position"].shift(1).fillna(0)
                            
                            # Strategy returns
                            beta_t = data["beta"].clip(lower=0.01)
                            wA = 1.0 / (1.0 + beta_t)
                            wB = 1.0 - wA
                            
                            pair_ret = np.where(
                                data["pos_lag"] == 1,
                                wA * data[f"ret_{pair_a}"] - wB * data[f"ret_{pair_b}"],
                                np.where(
                                    data["pos_lag"] == -1,
                                    -wA * data[f"ret_{pair_a}"] + wB * data[f"ret_{pair_b}"],
                                    0.0
                                )
                            )
                            
                            data["strategy"] = pair_ret
                            data["equity"] = (1.0 + data["strategy"]).cumprod()
                            
                            # Equity chart
                            eq_fig = go.Figure()
                            eq_fig.add_trace(go.Scatter(x=data.index, y=data["equity"].values, mode='lines', name='Strategy Equity', line=dict(color='#10B981', width=2)))
                            eq_fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
                            eq_fig.update_layout(height=350, title="Pairs Strategy Equity Curve", template="plotly_white")
                            st.plotly_chart(eq_fig, use_container_width=True)
                            
                            # Stats
                            total_ret = data["equity"].iloc[-1] - 1
                            trades = int((data["position"].diff().abs() > 0).sum())
                            time_in = (data["pos_lag"] != 0).mean() * 100
                            ann_vol = data["strategy"].std() * np.sqrt(252) * 100
                            sharpe = (data["strategy"].mean() * 252) / (data["strategy"].std() * np.sqrt(252)) if data["strategy"].std() > 0 else 0
                            max_dd = ((data["equity"] / data["equity"].cummax()) - 1).min() * 100
                            
                            st_c1, st_c2, st_c3, st_c4 = st.columns(4)
                            with st_c1:
                                st.metric("Total Return", f"{total_ret*100:.1f}%")
                            with st_c2:
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            with st_c3:
                                st.metric("Max Drawdown", f"{max_dd:.1f}%")
                            with st_c4:
                                st.metric("Trades", trades)
                            
                            st_c5, st_c6 = st.columns(2)
                            with st_c5:
                                st.metric("Annualized Vol", f"{ann_vol:.1f}%")
                            with st_c6:
                                st.metric("Time in Market", f"{time_in:.1f}%")
                            
                            # ========== CURRENT SIGNAL ==========
                            st.markdown("---")
                            st.markdown("### 🎯 Current Signal")
                            
                            last = data.iloc[-1]
                            sig_c1, sig_c2, sig_c3, sig_c4 = st.columns(4)
                            with sig_c1:
                                st.metric(f"{pair_a} Price", f"${last[pair_a]:.2f}")
                            with sig_c2:
                                st.metric(f"{pair_b} Price", f"${last[pair_b]:.2f}")
                            with sig_c3:
                                st.metric("Current Z-Score", f"{last['zscore']:.2f}")
                            with sig_c4:
                                pos_label = {1: f"Long {pair_a} / Short {pair_b}", -1: f"Short {pair_a} / Long {pair_b}", 0: "Flat"}
                                st.metric("Position", pos_label.get(int(last["position"]), "Flat"))
                            
                            # Summary
                            st.markdown("---")
                            meets_criteria = pvalue < 0.05 and (np.isfinite(half_life) and half_life < 60) and current_corr > 0.8
                            if meets_criteria:
                                st.success(f"✅ **{pair_a}/{pair_b} PASSES all filters**: Cointegrated (p={pvalue:.3f}), Half-life={half_life:.0f}d, Corr={current_corr:.2f}")
                            else:
                                issues = []
                                if pvalue >= 0.05:
                                    issues.append(f"Not cointegrated (p={pvalue:.3f})")
                                if not np.isfinite(half_life) or half_life >= 60:
                                    issues.append(f"Half-life too slow ({half_life:.0f}d)" if np.isfinite(half_life) else "No mean reversion")
                                if current_corr <= 0.8:
                                    issues.append(f"Low correlation ({current_corr:.2f})")
                                st.warning(f"⚠️ **{pair_a}/{pair_b} FAILS**: {', '.join(issues)}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
