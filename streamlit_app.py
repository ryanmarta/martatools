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
        
        mode = st.radio("Select Module", ["📺 Dashboard", "💎 Options", "🎯 Sniper", "🦅 Hunter"], label_visibility="collapsed")

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
            """Fetch news via RSS feeds - more reliable than yfinance API."""
            import re
            news_items = []
            
            # Yahoo Finance RSS feeds
            rss_urls = [
                ("https://finance.yahoo.com/rss/topstories", "Market"),
                ("https://finance.yahoo.com/rss/stock-market-news", "Stocks"),
            ]
            
            for url, category in rss_urls:
                try:
                    response = requests.get(url, timeout=5, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
                    })
                    if response.status_code == 200:
                        content = response.text
                        
                        # Simple XML parsing without external library
                        items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                        
                        for item_xml in items[:6]:
                            title_match = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>', item_xml)
                            link_match = re.search(r'<link>(.*?)</link>', item_xml)
                            
                            title = ""
                            if title_match:
                                title = title_match.group(1) or title_match.group(2) or ""
                            
                            link = link_match.group(1) if link_match else "https://finance.yahoo.com"
                            
                            if not title or len(title) < 10:
                                continue
                            
                            # Sentiment scoring
                            sentiment = 0
                            positive = ["surge", "jump", "rally", "gain", "rise", "soar", "beat", "record", "high", "bull", "growth", "profit", "buy", "upgrade", "boom"]
                            negative = ["fall", "drop", "crash", "plunge", "sink", "miss", "low", "bear", "loss", "fear", "sell", "cut", "down", "warn", "slump", "tumble"]
                            
                            title_lower = title.lower()
                            for word in positive:
                                if word in title_lower:
                                    sentiment += 1
                            for word in negative:
                                if word in title_lower:
                                    sentiment -= 1
                            
                            news_items.append({
                                "title": title.strip(),
                                "publisher": "Yahoo Finance",
                                "link": link.strip(),
                                "time": len(news_items),  # Use order as proxy
                                "sentiment": sentiment,
                                "ticker": category
                            })
                except Exception:
                    pass
            
            # Fallback: Try Google Finance RSS
            if len(news_items) < 3:
                try:
                    goog_url = "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"
                    response = requests.get(goog_url, timeout=5, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
                    })
                    if response.status_code == 200:
                        content = response.text
                        items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                        
                        for item_xml in items[:8]:
                            title_match = re.search(r'<title>(.*?)</title>', item_xml)
                            link_match = re.search(r'<link>(.*?)</link>', item_xml)
                            source_match = re.search(r'<source[^>]*>(.*?)</source>', item_xml)
                            
                            title = title_match.group(1) if title_match else ""
                            link = link_match.group(1) if link_match else "#"
                            source = source_match.group(1) if source_match else "Google News"
                            
                            if not title or len(title) < 10:
                                continue
                            
                            # Clean up title
                            title = re.sub(r'<[^>]+>', '', title).strip()
                            
                            sentiment = 0
                            title_lower = title.lower()
                            for word in ["surge", "jump", "rally", "gain", "rise", "beat", "record", "high", "growth"]:
                                if word in title_lower:
                                    sentiment += 1
                            for word in ["fall", "drop", "crash", "plunge", "miss", "low", "loss", "fear", "sell"]:
                                if word in title_lower:
                                    sentiment -= 1
                            
                            news_items.append({
                                "title": title,
                                "publisher": source,
                                "link": link,
                                "time": len(news_items),
                                "sentiment": sentiment,
                                "ticker": "NEWS"
                            })
                except Exception:
                    pass
            
            if not news_items:
                news_items = [{
                    "title": "Markets are active. Check Yahoo Finance for latest updates.",
                    "publisher": "Marta Tools",
                    "link": "https://finance.yahoo.com",
                    "time": 0,
                    "sentiment": 0,
                    "ticker": "INFO"
                }]
            
            return news_items[:10]

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
                bullish_count = sum(1 for d in indices_data.values() if d.get("change", 0) > 0 and "VIX" not in d.get("symbol", ""))
                total_indices = len([d for d in indices_data.values() if "VIX" not in d.get("symbol", "")])
                vix_data = indices_data.get("VIX (Fear)", {})
                vix_level = vix_data.get("price", 20)
                
                if vix_level > 30:
                    fear_level, fear_emoji = "FEAR", "😨"
                    fear_color = "#EF4444"
                elif vix_level > 20:
                    fear_level, fear_emoji = "CAUTION", "😐"
                    fear_color = "#F59E0B"
                else:
                    fear_level, fear_emoji = "GREED", "🤑"
                    fear_color = "#10B981"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {fear_color}15, {fear_color}30); 
                            padding: 24px; border-radius: 16px; border: 1px solid {fear_color}40; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">{fear_emoji}</div>
                    <h2 style="margin: 0; color: {fear_color}; font-size: 1.8rem; letter-spacing: 2px;">{fear_level}</h2>
                    <p style="margin: 12px 0 0 0; color: #64748B; font-size: 0.9rem;">
                        VIX: <strong>{vix_level:.1f}</strong> · {bullish_count}/{total_indices} indices up
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
                    "Iron Condor", "Iron Butterfly"
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
                        
                        for _, long_call in calls[calls["strike"] <= spot * 1.05].iterrows():
                            for _, short_call in calls[calls["strike"] > long_call["strike"]].iterrows():
                                if short_call["strike"] - long_call["strike"] > spot * 0.20:
                                    continue
                                
                                K_long = long_call["strike"]
                                K_short = short_call["strike"]
                                
                                # ========================================
                                # TRADER-FOCUSED CONSTRAINTS
                                # ========================================
                                
                                # 1. STRIKE PROXIMITY CONSTRAINT
                                if trader_mode_active:
                                    if not check_spread_proximity(K_long, K_short, spot, strike_proximity):
                                        continue  # Skip - both strikes too far from spot
                                
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
                        
                        for _, long_put in puts[puts["strike"] >= spot * 0.95].iterrows():
                            for _, short_put in puts[puts["strike"] < long_put["strike"]].iterrows():
                                if long_put["strike"] - short_put["strike"] > spot * 0.20:
                                    continue
                                
                                K_long = long_put["strike"]
                                K_short = short_put["strike"]
                                
                                # TRADER-FOCUSED: Strike proximity check
                                if trader_mode_active:
                                    if not check_spread_proximity(K_long, K_short, spot, strike_proximity):
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
                    
                    # ============ DISPLAY RESULTS (Capital Allocator Output) ============
                    if legs and trade_viable and trade_metrics["ev_dollars"] is not None and trade_metrics["ev_dollars"] > 0:
                        ev = trade_metrics["ev_dollars"]
                        ev_per_risk = trade_metrics["ev_per_dollar_risked"] or 0
                        prob_profit = trade_metrics["prob_profit"] or 0
                        prob_max_profit = trade_metrics.get("prob_max_profit", prob_profit)
                        prob_max_loss = trade_metrics.get("prob_max_loss", 0)
                        prob_conf = trade_metrics.get("prob_confidence", 0.7)
                        tail_eps = trade_metrics.get("tail_epsilon", 0.01)
                        is_deep_itm = trade_metrics.get("is_deep_itm", False)
                        quality = trade_metrics.get("quality_score", 0)
                        trade_label = trade_metrics.get("trade_label", "✅ ACCEPTABLE")
                        trade_sublabel = trade_metrics.get("trade_sublabel", "")
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
                            
                            st.info("""
                            **What this means:**
                            - Expected value is ≤ $0 after slippage
                            - Probability-weighted outcomes don't justify the trade
                            - Liquidity/execution costs consume theoretical edge
                            - Trade quality score below minimum threshold
                            
                            **Try:**
                            - Different expiration (front-month usually more liquid)
                            - Closer to ATM strikes
                            - Different strategy type
                            - Wait for better market conditions
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

        TICKER_LIST = [
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "TXN", "INTC", "IBM", "MU", "NOW", "UBER", "PANW",
            "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "AXP", "BLK", "C", "PYPL", "HOOD", "COIN", "SOFI",
            "WMT", "COST", "PG", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "TGT", "LOW", "TJX",
            "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "PFE", "AMGN", "ISRG", "BMY", "GILD", "CVS",
            "CAT", "DE", "HON", "GE", "UNP", "UPS", "BA", "LMT", "RTX", "XOM", "CVX", "COP", "SLB", "EOG",
            "MSTR", "MARA", "PLTR", "DKNG", "ROKU", "SQ", "AFRM", "RIOT", "CLSK", "CVNA", "UPST", "AI", "GME", "AMC",
            "SPY", "QQQ", "IWM", "DIA", "TLT",
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


if __name__ == "__main__":
    main()

