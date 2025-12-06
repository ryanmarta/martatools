import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
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
        st.title("Marta Tools 🛠️")
        st.caption("v1.0 | Quantitative Trading Suite")

        mode = st.radio("MODULE", ["📺 DASHBOARD", "💎 OPTIONIZER", "🎯 SNIPER", "🦅 THE HUNTER"])
        st.markdown("---")

        if mode == "🎯 SNIPER":
            ticker = st.text_input("TICKER", value="TSLA").upper()
            rf = st.number_input("RISK FREE (%)", value=4.5)
            cap = st.number_input("CAPITAL ($)", value=100000)
            kelly = st.slider("KELLY FACTOR", 0.1, 1.0, 0.5)

    # --- SHARED INIT (only for Sniper mode now) ---
    if mode == "🎯 SNIPER":
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
    if mode == "📺 DASHBOARD":
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
                except:
                    pass
            return data

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

        with st.spinner("Loading market data..."):
            indices_data = fetch_indices()
            news_data = fetch_news()

        # --- INDICES TICKER BAR ---
        if indices_data:
            cols = st.columns(len(indices_data))
            for i, (name, data) in enumerate(indices_data.items()):
                with cols[i]:
                    delta_color = "normal" if data["change"] >= 0 else "inverse"
                    if "VIX" in name:
                        delta_color = "inverse" if data["change"] >= 0 else "normal"
                        st.metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f}%", delta_color=delta_color)
                    else:
                        st.metric(name, f"${data['price']:.2f}", f"{data['change']:+.2f}%", delta_color=delta_color)
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
    elif mode == "💎 OPTIONIZER":
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

                dt = (datetime.strptime(sel_exp, "%Y-%m-%d").date() - date.today()).days
                T = max(dt / 365.0, 0.001)

                engine = PricingKernel(spot, sel_strike, T, rf / 100, sigma, sel_type)
                bs = engine.price_bsm()
                hes = engine.price_heston_proxy()
                mc = engine.price_mc(sims=50000)
                greeks = engine.get_greeks()

                consensus = (bs * 0.4) + (hes * 0.3) + (mc * 0.3)
                edge = consensus - mkt_p

                win = abs(greeks.delta)
                k_alloc = max(0, (win * 2 - (1 - win)) / 2) * kelly if edge > 0 else 0

                # Probability of Profit (break-even based, risk-neutral)
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
                    pop = np.nan

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
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=chain["strike"], y=chain["impliedVolatility"], mode="markers", name="Market"))
        if surface.valid:
            xs = np.linspace(chain["strike"].min(), chain["strike"].max(), 100)
            fig_s.add_trace(go.Scatter(x=xs, y=[surface.get_iv(x) for x in xs], mode="lines", name="Spline"))
        fig_s.update_layout(template="plotly_white", height=400, title="Volatility Skew")
        st.plotly_chart(fig_s, use_container_width=True)

    # ==========================================
    # MODULE C: SNIPER (Single Stock Deep Dive)
    # ==========================================
    elif mode == "🎯 SNIPER":
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
    elif mode == "🦅 THE HUNTER":
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

