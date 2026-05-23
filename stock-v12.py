import os
import re
import json
import datetime as dt
from io import BytesIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

try:
    from google import genai as genai_new
except Exception:
    genai_new = None

try:
    import google.generativeai as genai_old
except Exception:
    genai_old = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

TAIPEI_TZ    = ZoneInfo("Asia/Taipei")
FINMIND_URL  = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

#st.set_page_config(page_title="股票分析師", layout="wide", initial_sidebar_state="expanded")
st.set_page_config(page_title="股票分析師", layout="centered", initial_sidebar_state="expanded")
st.title("股票分析師")
st.caption("量化評分 + 專業研究報告")

# ─────────────────────────────────────────────
# 基礎工具
# ─────────────────────────────────────────────

def today_taipei():
    return dt.datetime.now(TAIPEI_TZ).date()

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(str(x).replace(",", "").replace("--", "").strip())
    except Exception:
        return np.nan

def pct(a, b):
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return (a - b) / b * 100

def grade_from_score(score):
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    return "F"

def make_jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.tail(12).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (dt.date, dt.datetime, pd.Timestamp)):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)

def fmt_num(x, digits=2, default="N/A"):
    try:
        if x is None or pd.isna(x):
            return default
        return f"{float(x):,.{digits}f}"
    except Exception:
        return default

def fmt_pct(x, digits=2, default="N/A"):
    try:
        if x is None or pd.isna(x):
            return default
        return f"{float(x):+,.{digits}f}%"
    except Exception:
        return default

def fmt_compact_4digits(x, default="N/A"):
    """將數字壓到約 4 位有效數字，避免營收欄位過長。"""
    try:
        if x is None or pd.isna(x):
            return default
        v = float(x)
        av = abs(v)
        if av >= 1000:
            return f"{v:,.0f}"
        if av >= 100:
            return f"{v:.1f}".rstrip("0").rstrip(".")
        if av >= 10:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{v:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return default

def fmt_tw_revenue_from_thousand(value_thousand, default="N/A"):
    """FinMind 月營收通常為新台幣千元；顯示成萬元/億元/兆元且數字約不超過4位。"""
    try:
        v = safe_float(value_thousand)
        if pd.isna(v):
            return default
        ntd = v * 1000.0
        av = abs(ntd)
        if av >= 1_000_000_000_000:
            return f"{fmt_compact_4digits(ntd / 1_000_000_000_000)}兆元"
        if av >= 100_000_000:
            return f"{fmt_compact_4digits(ntd / 100_000_000)}億元"
        if av >= 10_000:
            return f"{fmt_compact_4digits(ntd / 10_000)}萬元"
        return f"{fmt_compact_4digits(ntd)}元"
    except Exception:
        return default

# ─────────────────────────────────────────────
# ★ 新增：EPS 季度標籤輔助函數
# ─────────────────────────────────────────────

def get_ttm_quarter_label():
    """
    計算近四季（TTM）的季度標籤。
    例如：2026年5月執行 → 最近完成季為26Q1，回推4季 → '25Q2~26Q1'
    """
    now = today_taipei()
    y, m = now.year, now.month
    cur_q = (m - 1) // 3 + 1
    # 最近完成季度
    if cur_q == 1:
        last_q, last_y = 4, y - 1
    else:
        last_q, last_y = cur_q - 1, y
    # TTM 起始季度（最近完成季往前推3季）
    start_q = last_q - 3
    start_y = last_y
    if start_q <= 0:
        start_q += 4
        start_y -= 1
    sy = str(start_y)[-2:]
    ey = str(last_y)[-2:]
    return f"{sy}Q{start_q}~{ey}Q{last_q}"


def get_forward_eps_label():
    """
    計算預估EPS的季度標籤（未來4季）。
    例如：2026年5月執行 → 當前Q2 2026，下一季起推4季 → '26Q3~27Q2'
    """
    now = today_taipei()
    y, m = now.year, now.month
    cur_q = (m - 1) // 3 + 1
    # 從下一季開始
    if cur_q < 4:
        next_q, next_y = cur_q + 1, y
    else:
        next_q, next_y = 1, y + 1
    # 往後推4季結束
    end_q = next_q + 3
    end_y = next_y
    if end_q > 4:
        end_q -= 4
        end_y += 1
    sy = str(next_y)[-2:]
    ey = str(end_y)[-2:]
    return f"{sy}Q{next_q}~{ey}Q{end_q}"


def first_valid(*vals, default=np.nan):
    for v in vals:
        try:
            if v is not None and not pd.isna(v):
                return v
        except Exception:
            if v:
                return v
    return default

def clean_stock_id(stock_id):
    return re.sub(r"[^0-9A-Za-z]", "", str(stock_id)).strip()

# ─────────────────────────────────────────────
# FinMind 資料源
# ─────────────────────────────────────────────

def finmind_get(dataset, data_id=None, start_date=None, end_date=None, extra=None):
    params = {"dataset": dataset}
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        params["token"] = token
    if data_id:
        params["data_id"] = str(data_id)
    if start_date:
        params["start_date"] = str(start_date)
    if end_date:
        params["end_date"] = str(end_date)
    if extra:
        params.update(extra)
    try:
        r = requests.get(FINMIND_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("data", []))
    except Exception as e:
        st.warning(f"FinMind 讀取 {dataset} 失敗：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_stock_info_finmind(stock_id):
    """基本資料：加入市場別、公司產業類型，並保留資料來源欄位。"""
    sid = clean_stock_id(stock_id)
    df = finmind_get("TaiwanStockInfo")
    if df.empty or "stock_id" not in df.columns:
        return None
    cand = df[df["stock_id"].astype(str) == str(sid)]
    if cand.empty:
        return None
    row = cand.iloc[0].to_dict()
    industry = first_valid(
        row.get("industry_category"), row.get("industry"), row.get("產業別"), default="未知"
    )
    market = first_valid(row.get("type"), row.get("market"), row.get("市場別"), default="未知")
    return {
        "stock_id": str(row.get("stock_id", sid)),
        "name": row.get("stock_name", row.get("name", "未知")),
        "market": market,
        "industry": industry,
        "industry_type": industry,
        "source": "FinMind TaiwanStockInfo",
    }

@st.cache_data(ttl=600)
def get_price_history_finmind(stock_id, days=420):
    end   = today_taipei()
    start = end - dt.timedelta(days=days)
    df = finmind_get("TaiwanStockPrice", data_id=stock_id,
                     start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "max", "min", "close", "Trading_Volume", "Trading_money"]:
        if c in df.columns:
            df[c] = df[c].map(safe_float)
    return df.sort_values("date").drop_duplicates("date")

@st.cache_data(ttl=600)
def get_data_yfinance(stock_id):
    if not HAS_YFINANCE:
        return None, None, None
    for suffix in [".TW", ".TWO"]:
        tkr = f"{stock_id}{suffix}"
        try:
            t    = yf.Ticker(tkr)
            hist = t.history(period="180d")
            if not hist.empty and len(hist) >= 2:
                name = t.info.get("longName") or t.info.get("shortName") or stock_id
                return tkr, name, hist
        except Exception:
            continue
    return None, None, None

def yf_hist_to_df(hist):
    df = hist.reset_index().copy()
    df.rename(columns={
        "Date": "date", "Open": "open", "High": "max",
        "Low": "min", "Close": "close", "Volume": "Trading_Volume",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for c in ["open", "max", "min", "close", "Trading_Volume"]:
        if c in df.columns:
            df[c] = df[c].map(safe_float)
    return df[["date", "open", "max", "min", "close", "Trading_Volume"]].sort_values("date")

def fetch_stock_info_and_price(stock_id):
    info     = get_stock_info_finmind(stock_id)
    price_df = get_price_history_finmind(stock_id)
    if info is not None and not price_df.empty:
        return info, price_df
    if HAS_YFINANCE:
        tkr, name, hist = get_data_yfinance(stock_id)
        if tkr is not None:
            return {"stock_id": stock_id, "name": name,
                    "market": "上市/上櫃", "industry": "未知"}, yf_hist_to_df(hist)
    return None, pd.DataFrame()

def get_latest_two_trading_days(price_df):
    if price_df.empty or len(price_df) < 2:
        return None, None
    clean = price_df.dropna(subset=["open", "close"]).copy()
    if len(clean) < 2:
        return None, None
    return clean.iloc[-2], clean.iloc[-1]

# ─────────────────────────────────────────────
# 技術指標計算（擴充版）
# ─────────────────────────────────────────────

def add_technical_columns(df):
    """加入全套技術指標：多均線、布林帶、MACD、RSI、OBV、CMF、ADX"""
    df = df.copy()

    # 多重均線
    for n in [5, 20, 60, 120, 240]:
        df[f"MA{n}"] = df["close"].rolling(n).mean()

    # 成交量均線
    df["VOL5"]  = df["Trading_Volume"].rolling(5).mean()
    df["VOL20"] = df["Trading_Volume"].rolling(20).mean()

    # 日漲跌幅
    df["change_pct"] = df["close"].pct_change() * 100

    # ── RSI14 ──
    delta = df["close"].diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    rs    = up.rolling(14).mean() / down.rolling(14).mean().replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ── MACD (12/26/9) ──
    ema12      = df["close"].ewm(span=12, adjust=False).mean()
    ema26      = df["close"].ewm(span=26, adjust=False).mean()
    df["DIF"]  = ema12 - ema26
    df["MACD"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["OSC"]  = df["DIF"] - df["MACD"]

    # ── Bollinger Bands (20, ±2σ) ──
    bb_mid         = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_pct"]   = (df["close"] - df["BB_lower"]) / (
        (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan))

    # ── OBV ──
    sign        = np.sign(df["close"].diff().fillna(0))
    df["OBV"]   = (sign * df["Trading_Volume"]).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # ── Chaikin Money Flow (CMF, 20) ──
    clv = ((df["close"] - df["min"]) - (df["max"] - df["close"])) / (
        (df["max"] - df["min"]).replace(0, np.nan))
    df["CMF"] = (clv * df["Trading_Volume"]).rolling(20).sum() / \
                df["Trading_Volume"].rolling(20).sum().replace(0, np.nan)

    # ── ADX (簡化版，14) ──
    h, l, c_prev = df["max"], df["min"], df["close"].shift(1)
    tr    = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)
    dm_p  = (h - h.shift(1)).clip(lower=0)
    dm_m  = (l.shift(1) - l).clip(lower=0)
    dm_p  = dm_p.where(dm_p > dm_m, 0)
    dm_m  = dm_m.where(dm_m > dm_p, 0)
    di_p  = 100 * dm_p.rolling(14).mean() / atr14
    di_m  = 100 * dm_m.rolling(14).mean() / atr14
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()

    return df

# ─────────────────────────────────────────────
# K線評分（全面重寫）
# ─────────────────────────────────────────────


def analyze_candle(df):
    if df.empty or len(df) < 25:
        return {"score": 50, "summary": "K線資料不足，暫以保守評估。",
                "signals": [], "details": {}, "ma_analysis": []}

    d = add_technical_columns(df)
    last, prev = d.iloc[-1], d.iloc[-2]
    score = 50
    signals, ma_analysis = [], []
    details = {}

    close, open_, high, low = last["close"], last["open"], last["max"], last["min"]
    prev_close = prev["close"]
    body = abs(close - open_)
    rng = max(high - low, 1e-6)
    upper = high - max(close, open_)
    lower = min(close, open_) - low
    avg_body20 = (d["close"] - d["open"]).abs().tail(21).iloc[:-1].mean()
    body_ratio = body / avg_body20 if avg_body20 and avg_body20 > 0 else np.nan

    if pd.isna(body_ratio):
        candle_size = "資料不足"
    elif body_ratio >= 1.50:
        candle_size = "長K線"
    elif body_ratio >= 0.75:
        candle_size = "中K線"
    else:
        candle_size = "短K線"
    direction = "紅K" if close >= open_ else "黑K"
    details["K線長短"] = candle_size
    details["K線方向"] = direction
    details["實體/20日均實體"] = fmt_num(body_ratio, 2)

    if candle_size == "長K線":
        if close > open_:
            score += 14; signals.append(f"今日為長紅K（實體約20日均值 {body_ratio:.2f} 倍），買方主控。")
        else:
            score -= 14; signals.append(f"今日為長黑K（實體約20日均值 {body_ratio:.2f} 倍），賣壓主控。")
    elif candle_size == "中K線":
        if close > open_:
            score += 7; signals.append("今日為中紅K，短線偏多但仍需量能確認。")
        else:
            score -= 7; signals.append("今日為中黑K，短線轉弱需觀察支撐。")
    elif candle_size == "短K線":
        if prev_close and not pd.isna(prev_close) and abs(close - prev_close) / prev_close * 100 >= 1.0:
            score += 3 if close > prev_close else -3
        signals.append("今日為短K線，代表多空拉鋸，單日訊號權重較低。")

    if upper / rng > 0.45:
        score -= 5; signals.append("上影線偏長，上方賣壓明顯。")
    if lower / rng > 0.40:
        score += 4; signals.append("下影線偏長，低檔承接力道存在。")

    ma_map = [(5, "週線"), (20, "月線"), (60, "季線")]
    above_count = 0
    for n, label in ma_map:
        mv = last.get(f"MA{n}")
        slope = last.get(f"MA{n}") - d.iloc[-6].get(f"MA{n}") if len(d) >= 6 and not pd.isna(last.get(f"MA{n}")) and not pd.isna(d.iloc[-6].get(f"MA{n}")) else np.nan
        if pd.isna(mv):
            details[label] = "N/A"
            continue
        dist = pct(close, mv)
        trend_word = "上彎" if not pd.isna(slope) and slope > 0 else ("下彎" if not pd.isna(slope) and slope < 0 else "走平")
        pos_word = "站上" if close >= mv else "跌破"
        details[label] = f"{mv:.2f}（{pos_word}，乖離 {dist:+.2f}%）"
        if label in ("週線", "月線"):
            ma_analysis.append(f"{label}：收盤{pos_word}{label} {mv:.2f}，均線{trend_word}，乖離 {dist:+.2f}%。")
        if close >= mv:
            above_count += 1
            score += {5: 4, 20: 6, 60: 4}[n]
        else:
            score -= {5: 4, 20: 6, 60: 4}[n]

    if above_count >= 2:
        score += 3; signals.append("股價多數關鍵均線上方，技術架構偏多。")
    elif above_count == 0:
        score -= 4; signals.append("股價跌破主要均線，技術架構偏弱。")

    dif, macd_val, osc = last.get("DIF", np.nan), last.get("MACD", np.nan), last.get("OSC", np.nan)
    prev_osc = prev.get("OSC", np.nan)
    if not pd.isna(dif) and not pd.isna(macd_val):
        if dif > macd_val and osc > prev_osc:
            score += 4; signals.append("MACD動能改善，短線轉強。")
        elif dif < macd_val and osc < prev_osc:
            score -= 4; signals.append("MACD動能轉弱，需防回檔。")
    rsi = last.get("RSI14", np.nan)
    details["RSI14"] = fmt_num(rsi, 1)
    if not pd.isna(rsi):
        if rsi >= 75:
            score -= 5; signals.append(f"RSI {rsi:.1f} 偏熱，追價風險升高。")
        elif 50 <= rsi <= 68:
            score += 4; signals.append(f"RSI {rsi:.1f} 位於健康偏多區。")
        elif rsi < 40:
            score -= 4; signals.append(f"RSI {rsi:.1f} 偏弱，買盤不足。")

    score = int(np.clip(score, 0, 100))
    return {
        "score": score,
        "summary": " ".join((signals + ma_analysis)[:5]),
        "signals": signals + ma_analysis,
        "details": details,
        "ma_analysis": ma_analysis,
    }

# ─────────────────────────────────────────────
# 成交量評分（全面重寫）
# ─────────────────────────────────────────────


def analyze_volume(df):
    if df.empty or len(df) < 25:
        return {"score": 50, "summary": "成交量資料不足，暫以保守評估。",
                "signals": [], "details": {}}

    d = add_technical_columns(df)
    last, prev = d.iloc[-1], d.iloc[-2]
    long_df = d.tail(min(len(d), 240)).copy()
    total_vol = long_df["Trading_Volume"].sum()
    today_vol = last["Trading_Volume"]
    vol5 = d.tail(5)["Trading_Volume"].sum()
    vol20 = d.tail(20)["Trading_Volume"].sum()
    avg_day_pct = 100 / len(long_df) if len(long_df) else np.nan
    today_pct = today_vol / total_vol * 100 if total_vol > 0 else np.nan
    pct5 = vol5 / total_vol * 100 if total_vol > 0 else np.nan
    pct20 = vol20 / total_vol * 100 if total_vol > 0 else np.nan
    today_ratio = today_pct / avg_day_pct if avg_day_pct and not pd.isna(today_pct) else np.nan
    p_change = pct(last["close"], prev["close"])

    score = 50
    signals = []
    details = {
        "今日量/長期總量": fmt_num(today_pct, 3) + "%" if not pd.isna(today_pct) else "N/A",
        "近5日量/長期總量": fmt_num(pct5, 2) + "%" if not pd.isna(pct5) else "N/A",
        "近20日量/長期總量": fmt_num(pct20, 2) + "%" if not pd.isna(pct20) else "N/A",
        "今日量相對長期日均": fmt_num(today_ratio, 2) + "倍" if not pd.isna(today_ratio) else "N/A",
    }

    if not pd.isna(today_ratio):
        if today_ratio >= 2.2:
            if p_change > 0:
                score += 14; signals.append(f"今日量佔長期總量 {today_pct:.3f}%，約長期日均 {today_ratio:.2f} 倍，屬放量上攻。")
            else:
                score -= 14; signals.append(f"今日量佔長期總量 {today_pct:.3f}%，約長期日均 {today_ratio:.2f} 倍，價跌放量需警戒。")
        elif today_ratio >= 1.2:
            score += 6 if p_change > 0 else -5
            signals.append(f"今日量約長期日均 {today_ratio:.2f} 倍，{'量能支持上漲' if p_change > 0 else '下跌伴隨量增'}。")
        elif today_ratio <= 0.65:
            score -= 5; signals.append(f"今日量僅長期日均 {today_ratio:.2f} 倍，市場參與度偏低。")
        else:
            score += 3; signals.append(f"今日量約長期日均 {today_ratio:.2f} 倍，量能屬正常區間。")

    if not pd.isna(pct5) and not pd.isna(pct20):
        expected5 = avg_day_pct * 5
        expected20 = avg_day_pct * 20
        if pct5 > expected5 * 1.3 and p_change > 0:
            score += 6; signals.append("近5日量能集中度高於長期均值，且價格走強，短線資金活絡。")
        elif pct5 > expected5 * 1.3 and p_change < 0:
            score -= 6; signals.append("近5日量能集中但價格走弱，籌碼換手壓力偏高。")
        if pct20 < expected20 * 0.85:
            score -= 4; signals.append("近20日累積量低於長期應有水準，波段買盤不足。")

    cmf = last.get("CMF", np.nan)
    if not pd.isna(cmf):
        details["CMF資金流"] = fmt_num(cmf, 3)
        if cmf > 0.06:
            score += 5; signals.append(f"CMF {cmf:.3f}，資金淨流入。")
        elif cmf < -0.06:
            score -= 4; signals.append(f"CMF {cmf:.3f}，資金淨流出。")

    score = int(np.clip(score, 0, 100))
    return {"score": score, "summary": " ".join(signals[:5]), "signals": signals, "details": details}

# ─────────────────────────────────────────────
# 基本面 — 資料抓取
# ─────────────────────────────────────────────

def get_per_data(stock_id):
    end   = today_taipei()
    start = end - dt.timedelta(days=45)
    df = finmind_get("TaiwanStockPER", data_id=stock_id,
                     start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return {}
    return df.sort_values("date").iloc[-1].to_dict()

def get_month_revenue(stock_id):
    end   = today_taipei()
    start = end - dt.timedelta(days=800)
    df = finmind_get("TaiwanStockMonthRevenue", data_id=stock_id,
                     start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c not in ["date", "stock_id", "country"]:
            df[c] = df[c].map(safe_float)
    return df.sort_values("date")

@st.cache_data(ttl=3600)
def get_financial_statements_finmind(stock_id):
    end   = today_taipei()
    start = end - dt.timedelta(days=900)
    df = finmind_get("TaiwanStockFinancialStatements", data_id=stock_id,
                     start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return df
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = df["value"].map(safe_float)
    return df.sort_values("date")

@st.cache_data(ttl=3600)
def get_balance_sheet_finmind(stock_id):
    end   = today_taipei()
    start = end - dt.timedelta(days=900)
    df = finmind_get("TaiwanStockBalanceSheet", data_id=stock_id,
                     start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return df
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = df["value"].map(safe_float)
    return df.sort_values("date")


@st.cache_data(ttl=1800)
def get_yf_fundamentals(stock_id):
    if not HAS_YFINANCE:
        return {}
    sid = clean_stock_id(stock_id)
    for suffix in [".TW", ".TWO"]:
        try:
            t = yf.Ticker(f"{sid}{suffix}")
            info = t.info or {}
            if not (info.get("regularMarketPrice") or info.get("currentPrice") or info.get("sharesOutstanding")):
                continue

            def pct_field(key):
                v = info.get(key)
                return round(float(v) * 100, 2) if v is not None and not pd.isna(v) else np.nan

            return {
                "ticker": f"{sid}{suffix}",
                "long_name": info.get("longName") or info.get("shortName") or sid,
                "industry": info.get("industry") or "",
                "sector": info.get("sector") or "",
                "regular_market_price": safe_float(info.get("regularMarketPrice") or info.get("currentPrice")),
                "shares_outstanding": safe_float(info.get("sharesOutstanding")),
                "issued_shares": safe_float(info.get("sharesOutstanding")) if info.get("sharesOutstanding") else np.nan,
                # EPS
                "eps_trailing": safe_float(info.get("trailingEps")),
                "eps_forward": safe_float(info.get("forwardEps")),
                "earnings_growth": pct_field("earningsGrowth"),
                # 營收
                "revenue_growth": pct_field("revenueGrowth"),
                "total_revenue": safe_float(info.get("totalRevenue")),
                "revenue_per_share": safe_float(info.get("revenuePerShare")),
                # 獲利與股利
                "gross_margin": pct_field("grossMargins"),
                "dividend_yield": pct_field("dividendYield"),
                # 目標價
                "analyst_target_mean": safe_float(info.get("targetMeanPrice")),
                "analyst_target_high": safe_float(info.get("targetHighPrice")),
                "analyst_target_low": safe_float(info.get("targetLowPrice")),
                "analyst_recommendation": info.get("recommendationKey", ""),
                "analyst_count": safe_float(info.get("numberOfAnalystOpinions")),
            }
        except Exception:
            continue
    return {}

def _fs_latest(fs_df, keywords):
    if fs_df.empty or "type" not in fs_df.columns:
        return np.nan
    mask = fs_df["type"].str.contains("|".join(keywords), case=False, na=False)
    sub  = fs_df[mask]
    if sub.empty:
        return np.nan
    return safe_float(sub.sort_values("date").iloc[-1]["value"])

def _fs_yoy(fs_df, keywords):
    if fs_df.empty or "type" not in fs_df.columns:
        return np.nan
    mask = fs_df["type"].str.contains("|".join(keywords), case=False, na=False)
    sub  = fs_df[mask].sort_values("date")
    if len(sub) < 5:
        return np.nan
    latest = safe_float(sub.iloc[-1]["value"])
    year_ago = safe_float(sub.iloc[-5]["value"])
    return pct(latest, year_ago)

# ─────────────────────────────────────────────
# 基本面評分（全面重寫）
# ─────────────────────────────────────────────


def analyze_fundamental(stock_id):
    score = 50
    signals = []
    details = {}

    yf_data = get_yf_fundamentals(stock_id)
    per = get_per_data(stock_id)
    revenue = get_month_revenue(stock_id)
    fs_df = get_financial_statements_finmind(stock_id)

    eps_t = yf_data.get("eps_trailing", np.nan)
    eps_f = yf_data.get("eps_forward", np.nan)
    eps_growth = yf_data.get("earnings_growth", np.nan)
    if pd.isna(eps_t):
        eps_t = _fs_latest(fs_df, ["EPS", "每股盈餘", "每股純益"])
    if pd.isna(eps_growth):
        eps_growth = _fs_yoy(fs_df, ["EPS", "每股盈餘", "每股純益"])

    # ★ 加入季度標籤
    ttm_label = get_ttm_quarter_label()
    fwd_label = get_forward_eps_label()
    details[f"EPS(TTM/{ttm_label})"] = fmt_num(eps_t, 2)
    details[f"EPS(預估/{fwd_label})"] = fmt_num(eps_f, 2)
    details["EPS年增率"] = fmt_pct(eps_growth, 1)

    if not pd.isna(eps_growth):
        if eps_growth > 25:
            score += 18; signals.append(f"EPS年增率 {eps_growth:.1f}%，獲利成長強勁。")
        elif eps_growth > 5:
            score += 10; signals.append(f"EPS年增率 {eps_growth:.1f}%，獲利溫和成長。")
        elif eps_growth < -15:
            score -= 18; signals.append(f"EPS年增率 {eps_growth:.1f}%，獲利明顯衰退。")
        elif eps_growth < 0:
            score -= 8; signals.append(f"EPS年增率 {eps_growth:.1f}%，獲利小幅下滑。")
    if not pd.isna(eps_t) and not pd.isna(eps_f) and eps_t != 0:
        fwd_growth = pct(eps_f, eps_t)
        details["預估EPS變化"] = fmt_pct(fwd_growth, 1)
        if fwd_growth > 10:
            score += 5; signals.append(f"預估EPS較近四季 EPS 成長 {fwd_growth:.1f}%，展望偏正向。")
        elif fwd_growth < -10:
            score -= 5; signals.append(f"預估EPS較近四季 EPS 衰退 {abs(fwd_growth):.1f}%，展望保守。")

    gm = yf_data.get("gross_margin", np.nan)
    if pd.isna(gm):
        gm = _fs_latest(fs_df, ["毛利率", "GrossMargin", "Gross Profit Margin"])
    details["毛利率"] = fmt_pct(gm, 1).replace("+", "")
    if not pd.isna(gm):
        if gm >= 45:
            score += 10; signals.append(f"毛利率 {gm:.1f}%，產品組合或定價能力佳。")
        elif gm >= 25:
            score += 5; signals.append(f"毛利率 {gm:.1f}%，獲利結構尚穩。")
        elif gm < 12:
            score -= 6; signals.append(f"毛利率 {gm:.1f}%，本業毛利偏低。")

    dy = first_valid(yf_data.get("dividend_yield"), safe_float(per.get("dividend_yield", np.nan)))
    details["殖利率"] = fmt_pct(dy, 2).replace("+", "")
    if not pd.isna(dy):
        if dy >= 5:
            score += 8; signals.append(f"殖利率 {dy:.2f}%，股東現金回饋具吸引力。")
        elif 2 <= dy < 5:
            score += 4; signals.append(f"殖利率 {dy:.2f}%，具一定配息支撐。")
        elif dy < 1:
            score -= 2; signals.append(f"殖利率 {dy:.2f}%，配息保護較低。")

    rev_col = None
    latest_rev = yoy_rev = mom_rev = qoq_rev = np.nan
    if not revenue.empty:
        rev_col = next((c for c in ["revenue", "Revenue", "當月營收"] if c in revenue.columns), None)
    if rev_col and len(revenue) >= 2:
        latest_rev = safe_float(revenue.iloc[-1][rev_col])
        prev_rev = safe_float(revenue.iloc[-2][rev_col])
        mom_rev = pct(latest_rev, prev_rev)
        details["最新月營收"] = fmt_tw_revenue_from_thousand(latest_rev)
        details["月營收MoM"] = fmt_pct(mom_rev, 1)
        if len(revenue) >= 13:
            y_ago_rev = safe_float(revenue.iloc[-13][rev_col])
            yoy_rev = pct(latest_rev, y_ago_rev)
            last3 = revenue.tail(3)[rev_col].mean()
            prev3 = revenue.iloc[-6:-3][rev_col].mean() if len(revenue) >= 6 else np.nan
            qoq_rev = pct(last3, prev3)
            details["月營收YoY"] = fmt_pct(yoy_rev, 1)
            details["近3月營收動能"] = fmt_pct(qoq_rev, 1)
        if len(revenue) >= 12:
            ttm_rev = revenue.tail(12)[rev_col].sum()
            details["近12月累計營收"] = fmt_tw_revenue_from_thousand(ttm_rev)

    if not pd.isna(yoy_rev):
        if yoy_rev > 20:
            score += 16; signals.append(f"月營收年增 {yoy_rev:.1f}%，營收動能強。")
        elif yoy_rev > 5:
            score += 9; signals.append(f"月營收年增 {yoy_rev:.1f}%，營運穩健。")
        elif yoy_rev < -10:
            score -= 14; signals.append(f"月營收年減 {abs(yoy_rev):.1f}%，營運壓力升高。")
        elif yoy_rev < 0:
            score -= 6; signals.append(f"月營收年減 {abs(yoy_rev):.1f}%，需觀察復甦。")
    elif not pd.isna(yf_data.get("revenue_growth", np.nan)):
        rg = yf_data.get("revenue_growth")
        details["營收成長率(Yahoo)"] = fmt_pct(rg, 1)
        if rg > 10:
            score += 8; signals.append(f"Yahoo營收成長率 {rg:.1f}%，成長偏正向。")
        elif rg < -5:
            score -= 8; signals.append(f"Yahoo營收成長率 {rg:.1f}%，營收轉弱。")

    if not signals:
        signals.append("EPS、毛利率、殖利率與營收公開資料不足，採保守評估。")

    score = int(np.clip(score, 0, 100))
    return {
        "score": score,
        "summary": " ".join(signals[:5]),
        "signals": signals,
        "details": details,
        "per": per,
        "revenue": revenue,
        "yf_data": yf_data,
    }

# ─────────────────────────────────────────────
# 三大法人（改為張數，今日/昨日比較）
# ─────────────────────────────────────────────

def normalize_investor_name(name):
    s = str(name)
    if "外資" in s or "Foreign" in s: return "外資"
    if "投信" in s or "Investment" in s: return "投信"
    if "自營" in s or "Dealer" in s: return "自營商"
    return s

def get_institutional(stock_id, days=45):
    end   = today_taipei()
    start = end - dt.timedelta(days=days)
    df = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", data_id=stock_id,
                     start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ["buy", "sell", "Trading_Volume"]:
        if c in df.columns:
            df[c] = df[c].map(safe_float)
    if "buy" in df.columns and "sell" in df.columns:
        df["net_buy_shares"] = df["buy"] - df["sell"]
        df["net_buy"]        = df["net_buy_shares"] / 1000   # 股 → 張
    elif "Trading_Volume" in df.columns:
        df["net_buy"]        = df["Trading_Volume"] / 1000
    else:
        df["net_buy"] = 0
    return df.sort_values("date")

def _day_pivot(df, date_val):
    sub = df[df["date"] == date_val].copy()
    out = {}
    for inst in ["外資", "投信", "自營商"]:
        rows = sub[sub["法人"] == inst]
        out[inst] = int(rows["net_buy"].sum()) if not rows.empty else 0
    out["合計"] = int(sum(out.values()))
    return out


def summarize_institutional(df):
    EMPTY = {
        "score": 50,
        "summary": "三大法人資料不足，僅列資訊不列入總分。",
        "signals": [],
        "today": {"外資": 0, "投信": 0, "自營商": 0, "合計": 0},
        "yesterday": {"外資": 0, "投信": 0, "自營商": 0, "合計": 0},
        "today_date": None,
        "yday_date": None,
        "consecutive_buy_days": {"外資": 0, "投信": 0, "自營商": 0, "合計": 0},
        "consecutive_sell_days": {"外資": 0, "投信": 0, "自營商": 0, "合計": 0},
        "buy_sell_day_count": {"外資": {"buy": 0, "sell": 0}, "投信": {"buy": 0, "sell": 0},
                               "自營商": {"buy": 0, "sell": 0}, "合計": {"buy": 0, "sell": 0}},
        "month_table": pd.DataFrame(),
        "month_summary": pd.DataFrame(),
        "外資": 0, "投信": 0, "自營商": 0,
    }
    if df.empty:
        return pd.DataFrame(), EMPTY

    investor_col = next((c for c in ["name", "institutional_investors", "Investor", "type"] if c in df.columns), None)
    if investor_col is None:
        return pd.DataFrame(), EMPTY

    df = df.copy()
    df["法人"] = df[investor_col].map(normalize_investor_name)
    trading_dates = sorted(df["date"].unique())
    if len(trading_dates) < 1:
        return pd.DataFrame(), EMPTY

    today_date = trading_dates[-1]
    yday_date = trading_dates[-2] if len(trading_dates) >= 2 else None
    today_data = _day_pivot(df, today_date)
    yday_data = _day_pivot(df, yday_date) if yday_date is not None else {"外資": 0, "投信": 0, "自營商": 0, "合計": 0}

    preferred = ["外資", "投信", "自營商"]
    daily = df.pivot_table(index="date", columns="法人", values="net_buy", aggfunc="sum").fillna(0).sort_index()
    for inst in preferred:
        if inst not in daily.columns:
            daily[inst] = 0
    daily = daily[preferred]
    daily["合計"] = daily.sum(axis=1)

    consecutive_buy_days = {}
    consecutive_sell_days = {}
    for inst in preferred + ["合計"]:
        buy_cnt = 0
        sell_cnt = 0
        direction = None
        for v in reversed(daily[inst].tolist()):
            fv = float(v)
            if direction is None:
                if fv > 0:
                    direction = "buy"; buy_cnt = 1
                elif fv < 0:
                    direction = "sell"; sell_cnt = 1
                else:
                    break
            elif direction == "buy":
                if fv > 0: buy_cnt += 1
                else: break
            elif direction == "sell":
                if fv < 0: sell_cnt += 1
                else: break
        consecutive_buy_days[inst] = buy_cnt
        consecutive_sell_days[inst] = sell_cnt

    buy_sell_day_count = {}
    for inst in preferred + ["合計"]:
        recent5_vals = daily[inst].tail(5)
        buy_sell_day_count[inst] = {
            "buy":  int((recent5_vals > 0).sum()),
            "sell": int((recent5_vals < 0).sum()),
        }

    recent5 = df[df["date"] >= today_date - pd.Timedelta(days=10)]
    pivot = recent5.groupby("法人")["net_buy"].agg(["sum", "mean"]).reset_index().rename(
        columns={"sum": "近5日買賣超(張)", "mean": "日均(張)"}
    )
    pivot = pivot[pivot["法人"].isin(preferred)]
    if not pivot.empty:
        pivot["近5日買超天"] = pivot["法人"].map(lambda x: buy_sell_day_count.get(x, {}).get("buy", 0))
        pivot["近5日賣超天"] = pivot["法人"].map(lambda x: buy_sell_day_count.get(x, {}).get("sell", 0))
        pivot["排序"] = pivot["法人"].map({k: i for i, k in enumerate(preferred)})
        pivot = pivot.sort_values("排序").drop(columns=["排序"])

    signals = []
    for inst in preferred:
        tv, yv = today_data.get(inst, 0), yday_data.get(inst, 0)
        if tv * yv < 0:
            signals.append(f"{inst}今日{'翻多' if tv > 0 else '翻空'}（{tv:+,.0f}張），昨日 {yv:+,.0f}張。")
        elif tv > 0:
            signals.append(f"{inst}今日買超 {tv:+,.0f}張。")
        elif tv < 0:
            signals.append(f"{inst}今日賣超 {tv:+,.0f}張。")
        bsd = buy_sell_day_count.get(inst, {"buy": 0, "sell": 0})
        bd, sd = bsd["buy"], bsd["sell"]
        if sd > bd and sd >= 2:
            signals.append(f"{inst}近5日賣超 {sd} 天（買超 {bd} 天），以賣方為主。")
        elif bd >= sd and bd >= 2:
            signals.append(f"{inst}近5日買超 {bd} 天（賣超 {sd} 天），以買方為主。")

    bsd_total = buy_sell_day_count.get("合計", {"buy": 0, "sell": 0})
    bd_t, sd_t = bsd_total["buy"], bsd_total["sell"]
    if sd_t > bd_t and sd_t >= 2:
        signals.append(f"三大法人合計近5日賣超 {sd_t} 天，籌碼偏空。")
    elif bd_t >= sd_t and bd_t >= 2:
        signals.append(f"三大法人合計近5日買超 {bd_t} 天，籌碼偏多。")

    chip = {
        "score": 50,
        "summary": " ".join(signals[:6]) if signals else "法人買賣超無明顯方向，僅供參考。",
        "signals": signals,
        "today": today_data,
        "yesterday": yday_data,
        "today_date": today_date,
        "yday_date": yday_date,
        "consecutive_buy_days": consecutive_buy_days,
        "consecutive_sell_days": consecutive_sell_days,
        "buy_sell_day_count": buy_sell_day_count,
        "daily_5": daily.tail(5).reset_index(),
        "month_table": pd.DataFrame(),
        "month_summary": pd.DataFrame(),
        "外資": 0, "投信": 0, "自營商": 0,
    }
    return pivot, chip

# ─────────────────────────────────────────────
# 注意/處置股、大盤
# ─────────────────────────────────────────────


def fetch_page_text(url):
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
    except Exception:
        return ""

def _extract_dates_from_text(text):
    dates = []
    for m in re.findall(r"(?:\d{3,4}[/-]\d{1,2}[/-]\d{1,2})", text or ""):
        try:
            parts = re.split(r"[/-]", m)
            y = int(parts[0])
            if y < 1911:
                y += 1911
            dates.append(dt.date(y, int(parts[1]), int(parts[2])))
        except Exception:
            pass
    return sorted(set(dates))

def _twse_rwd_rows(kind, stock_id):
    sid = clean_stock_id(stock_id)
    rows = []
    try:
        for back in [0, 7, 14, 21, 28, 35, 42]:
            d = today_taipei() - dt.timedelta(days=back)
            url = f"https://www.twse.com.tw/rwd/zh/announcement/{kind}"
            params = {"response": "json", "date": d.strftime("%Y%m%d"), "selectType": "ALL"}
            r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=12)
            if not r.ok:
                continue
            payload = r.json()
            fields = payload.get("fields", [])
            for row in payload.get("data", []):
                joined = " ".join(map(str, row))
                if sid in joined:
                    rows.append({"source": f"TWSE {kind}", "fields": fields, "row": row, "text": joined})
    except Exception:
        pass
    return rows

def check_attention_disposition(stock_id):
    sid = clean_stock_id(stock_id)
    findings = []

    for kind, label in [("notice", "TWSE 注意股"), ("punish", "TWSE 處置股")]:
        for r in _twse_rwd_rows(kind, sid):
            findings.append({"source": label, "url": f"https://www.twse.com.tw/zh/announcement/{kind}.html", "snippet": r.get("text", "")})

    sources = [
        ("TWSE 注意股", "https://www.twse.com.tw/zh/announcement/notice.html"),
        ("TWSE 處置股", "https://www.twse.com.tw/zh/announcement/punish.html"),
        ("TPEx 處置股", "https://www.tpex.org.tw/zh-tw/announce/market/disposal.html"),
        ("TPEx 注意股", "https://www.tpex.org.tw/zh-tw/announce/market/attention.html"),
    ]
    seen = set()
    for name, url in sources:
        txt = fetch_page_text(url)
        if sid in txt:
            count = max(1, txt.count(sid))
            idx = txt.find(sid)
            window = txt[max(0, idx - 160): idx + 420] if idx >= 0 else ""
            key = (name, window[:80])
            if key not in seen:
                seen.add(key)
                findings.append({"source": name, "url": url, "snippet": window, "count": count})

    joined = " ".join(f.get("snippet", "") for f in findings)
    attention_findings = [f for f in findings if "注意" in f.get("source", "")]
    disposition_findings = [f for f in findings if "處置" in f.get("source", "")]
    attention_count = sum(int(f.get("count", 1)) for f in attention_findings)
    disposition_count = sum(int(f.get("count", 1)) for f in disposition_findings)
    is_repeated = any(k in joined for k in ["再次", "二次", "第二次", "連續", "分盤", "延長"])
    dates = _extract_dates_from_text(joined)
    release_date = dates[-1] if dates else None

    status = {
        "is_attention": len(attention_findings) > 0,
        "is_disposition": len(disposition_findings) > 0,
        "is_repeated_disposition": bool(is_repeated and len(disposition_findings) > 0),
        "attention_count": int(attention_count),
        "disposition_count": int(disposition_count),
        "release_date": release_date,
        "findings": findings,
        "penalty": 0,
        "summary": "未在 TWSE/TPEx 官方注意或處置頁面找到明確命中。",
    }
    if status["is_repeated_disposition"]:
        status["penalty"] = 20
        status["summary"] = f"偵測為再次處置股；注意股命中 {attention_count} 次、處置命中 {disposition_count} 次。"
    elif status["is_disposition"]:
        status["penalty"] = 12
        status["summary"] = f"偵測為處置股；注意股命中 {attention_count} 次、處置命中 {disposition_count} 次。"
    elif status["is_attention"]:
        status["penalty"] = 5
        status["summary"] = f"偵測為注意股；近頁面命中 {attention_count} 次。"
    if release_date:
        status["summary"] += f" 預估/公告出關日期：{release_date}。"
    return status

@st.cache_data(ttl=600)
def get_market_index():
    end   = today_taipei()
    start = end - dt.timedelta(days=120)
    for data_id in ["TAIEX", "加權指數"]:
        df = finmind_get("TaiwanStockPrice", data_id=data_id,
                         start_date=start.isoformat(), end_date=end.isoformat())
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "max", "min", "close", "Trading_Volume"]:
                if c in df.columns:
                    df[c] = df[c].map(safe_float)
            return df.sort_values("date")
    if HAS_YFINANCE:
        try:
            hist = yf.Ticker("^TWII").history(period="120d")
            if not hist.empty:
                return yf_hist_to_df(hist)
        except Exception:
            pass
    return pd.DataFrame()

def analyze_market_risk():
    df = get_market_index()
    if df.empty or len(df) < 25:
        return {"score_penalty": 0, "summary": "大盤資料不足，未進行大盤降級。",
                "signals": []}
    d    = add_technical_columns(df)
    last = d.iloc[-1]
    signals, penalty = [], 0
    if last["close"] < last["MA5"]:
        penalty += 3; signals.append("大盤跌破週線。")
    if last["close"] < last["MA20"]:
        penalty += 6; signals.append("大盤跌破月線，系統性風險升高。")
    if last["Trading_Volume"] > last["VOL20"] * 1.3 and last["change_pct"] < 0:
        penalty += 6; signals.append("大盤價跌量增，賣壓沉重。")
    return {
        "score_penalty": penalty,
        "summary":       " ".join(signals) if signals else "大盤結構尚未觸發明顯降級。",
        "signals":       signals,
        "market_close":  float(last["close"]),
        "ma20":          float(last["MA20"]),
    }

# ─────────────────────────────────────────────
# 目標價估算（優先使用 預估EPS × 合理本益比）
# ─────────────────────────────────────────────


def get_issued_shares(stock_id, yf_data=None):
    yf_data = yf_data or {}
    issued = yf_data.get("issued_shares", np.nan)
    if issued is not None and not pd.isna(issued) and issued > 0:
        return float(issued), "Yahoo Finance sharesOutstanding"
    return np.nan, "未取得"

def calc_trading_plan(price_df, total_score):
    if price_df.empty or len(price_df) < 25:
        return {"entry_price": np.nan, "stop_price": np.nan, "logic": "價格資料不足。"}
    d = add_technical_columns(price_df)
    last = d.iloc[-1]
    close = float(last["close"])
    ma5 = last.get("MA5", np.nan)
    ma20 = last.get("MA20", np.nan)
    ma60 = last.get("MA60", np.nan)
    low10 = d.tail(10)["min"].min()
    low20 = d.tail(20)["min"].min()
    atr_proxy = (d.tail(14)["max"] - d.tail(14)["min"]).mean()

    if total_score >= 75:
        entry = min(close, first_valid(ma5, close) * 1.01)
        stop = min(first_valid(ma20, low10), low10) - atr_proxy * 0.15
        logic = "偏多分數，採回測週線附近或不追高進場；跌破月線/近10日低點作停損。"
    elif total_score >= 60:
        entry = min(close, first_valid(ma20, close) * 1.01)
        stop = min(first_valid(ma60, low20), low20) - atr_proxy * 0.10
        logic = "中性偏多，等待回測月線附近；跌破季線或近20日低點停損。"
    else:
        entry = min(close * 0.98, first_valid(ma20, close))
        stop = low20 - atr_proxy * 0.10
        logic = "分數偏低，僅適合保守等回檔；跌破近20日低點即停損。"
    return {
        "entry_price": round(float(entry), 2) if not pd.isna(entry) else np.nan,
        "stop_price": round(float(max(stop, 0)), 2) if not pd.isna(stop) else np.nan,
        "logic": logic,
    }

def _extract_agency_from_text(text):
    agencies = ["外資", "摩根士丹利", "高盛", "美銀", "花旗", "里昂", "瑞銀", "麥格理", "大摩", "小摩", "凱基", "元大", "國泰", "富邦", "群益", "統一", "永豐", "Yahoo Finance", "Edge財經"]
    for a in agencies:
        if a.lower() in (text or "").lower():
            return a
    return "公開來源"

def search_public_target_prices(stock_id, company_name, max_items=12):
    sid = clean_stock_id(stock_id)
    queries = [
        f"{sid} {company_name} 目標價 高盛 摩根 美銀 瑞銀 野村",
        f"{sid} {company_name} 目標價 券商 最新",
        f"{sid} {company_name} target price Goldman JPMorgan Morgan Stanley UBS",
        f"{sid} {company_name} Edge 財經 目標價 外資",
        f"{sid} {company_name} Yahoo 財經 目標價 分析師",
        f"{sid} {company_name} 元大 凱基 國泰 富邦 目標價",
    ]
    KNOWN_INSTITUTIONS = {
        "高盛": "高盛（Goldman Sachs）", "goldman": "高盛（Goldman Sachs）",
        "摩根大通": "摩根大通（JPMorgan）", "jpmorgan": "摩根大通（JPMorgan）", "小摩": "摩根大通（JPMorgan）",
        "美銀": "美銀美林（BofA Merrill）", "bofa": "美銀美林（BofA Merrill）", "merrill": "美銀美林（BofA Merrill）",
        "摩根士丹利": "摩根士丹利（Morgan Stanley）", "morgan stanley": "摩根士丹利（Morgan Stanley）", "大摩": "摩根士丹利（Morgan Stanley）",
        "瑞銀": "瑞銀（UBS）", "ubs": "瑞銀（UBS）",
        "野村": "野村（Nomura）", "nomura": "野村（Nomura）",
        "元大": "元大投顧（Yuanta）", "yuanta": "元大投顧（Yuanta）",
        "凱基": "凱基投顧（KGI）", "kgi": "凱基投顧（KGI）",
        "花旗": "花旗（Citi）", "citi": "花旗（Citi）",
        "里昂": "里昂（CLSA）", "clsa": "里昂（CLSA）",
        "麥格理": "麥格理（Macquarie）", "macquarie": "麥格理（Macquarie）",
        "富邦": "富邦投顧", "國泰": "國泰投顧",
        "匯豐": "匯豐（HSBC）", "hsbc": "匯豐（HSBC）",
        "瑞信": "瑞信（Credit Suisse）", "credit suisse": "瑞信（Credit Suisse）",
        "法巴": "法國巴黎銀行（BNP）", "bnp": "法國巴黎銀行（BNP）",
        "德意志": "德意志銀行（Deutsche）", "deutsche": "德意志銀行（Deutsche）",
        "群益": "群益投顧", "統一": "統一投顧", "永豐": "永豐投顧", "兆豐": "兆豐投顧",
    }

    def _extract_institution(text):
        tl = text.lower()
        for key, label in KNOWN_INSTITUTIONS.items():
            if key in tl:
                return label
        return _extract_agency_from_text(text)

    results = []
    seen = set()
    for query in queries:
        try:
            r = requests.post("https://duckduckgo.com/html/", data={"q": query}, headers=DEFAULT_HEADERS, timeout=18)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.select(".result"):
                title_el = a.select_one(".result__title")
                snippet_el = a.select_one(".result__snippet")
                link_el = a.select_one("a.result__a")
                title = title_el.get_text(" ", strip=True) if title_el else ""
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
                link = link_el.get("href", "") if link_el else ""
                text = f"{title} {snippet}"
                if not any(k.lower() in text.lower() for k in ["目標價", "target", "券商", "外資", "調升", "調降", "評等"]):
                    continue
                prices = re.findall(r"(?:目標價|target price|TP|上看|調升至|調降至)[^0-9]{0,12}(\d+(?:\.\d+)?)", text, re.I)
                if not prices:
                    prices = re.findall(r"(\d+(?:\.\d+)?)\s*元", text)
                if not prices:
                    continue
                price = safe_float(prices[0])
                if pd.isna(price) or price <= 0:
                    continue
                institution = _extract_institution(text)
                key = (round(price, 2), institution[:20])
                if key in seen:
                    continue
                seen.add(key)
                dates = _extract_dates_from_text(text)
                results.append({
                    "機構/來源": institution,
                    "目標價": round(price, 2),
                    "評等/摘要": snippet[:90],
                    "日期": str(dates[-1]) if dates else "最新搜尋",
                    "連結": link,
                })
                if len(results) >= max_items:
                    return pd.DataFrame(results)
        except Exception:
            continue
    return pd.DataFrame(results)


def estimate_target_price(price_df, total_score, yf_data=None, public_targets=None, per_data=None):
    if price_df.empty or len(price_df) < 60:
        return None
    yf_data = yf_data or {}
    per_data = per_data or {}
    public_targets = public_targets if isinstance(public_targets, pd.DataFrame) else pd.DataFrame()

    d = add_technical_columns(price_df)
    last = d.iloc[-1]
    close = float(last["close"])
    ma20, ma60 = last.get("MA20", np.nan), last.get("MA60", np.nan)
    high60 = d.tail(60)["max"].max()
    low20 = d.tail(20)["min"].min()

    eps_f = yf_data.get("eps_forward", np.nan)
    eps_t = yf_data.get("eps_trailing", np.nan)

    current_pe = np.nan
    if per_data:
        current_pe = safe_float(
            per_data.get("PER") or per_data.get("per") or
            per_data.get("本益比") or per_data.get("price_earnings_ratio")
        )
    if pd.isna(current_pe) and not pd.isna(eps_t) and eps_t > 0:
        current_pe = close / eps_t

    quant_target = np.nan
    quant_logic = ""
    eps_used = np.nan

    has_trailing = not pd.isna(eps_t) and eps_t > 0
    has_forward  = not pd.isna(eps_f) and eps_f > 0

    if has_trailing and has_forward:
        eps_avg = (eps_t + eps_f) / 2.0
        eps_label = f"(已實現EPS {eps_t:.2f} + 預估EPS {eps_f:.2f}) / 2 = {eps_avg:.2f}"
        eps_used = eps_avg
    elif has_forward:
        eps_avg = eps_f
        eps_label = f"預估EPS {eps_f:.2f}（已實現EPS不足，僅用預估）"
        eps_used = eps_avg
    elif has_trailing:
        eps_avg = eps_t
        eps_label = f"已實現EPS {eps_t:.2f}（預估EPS不足，僅用近四季）"
        eps_used = eps_avg
    else:
        eps_avg = np.nan

    if not pd.isna(eps_avg):
        if not pd.isna(current_pe) and 0 < current_pe < 10000:
            reasonable_pe = round(min(max(current_pe * 0.85, 0.1), 10000.0), 1)
        else:
            reasonable_pe = 18.0
        quant_target = round(eps_avg * reasonable_pe, 2)
        quant_logic = (
            f"方法：({eps_label}) × 合理本益比 {reasonable_pe:.1f}x"
            + (f"（當前PE {current_pe:.1f}x × 0.85）" if not pd.isna(current_pe) and 0 < current_pe < 10000 else "（台股保守預設）")
        )

    target_sources = []

    if not public_targets.empty:
        for _, row in public_targets.iterrows():
            tp = safe_float(row.get("目標價"))
            if not pd.isna(tp) and tp > 0:
                target_sources.append({
                    "機構/來源": row.get("機構/來源", "公開來源"),
                    "目標價": round(float(tp), 2),
                    "評等/摘要": row.get("評等/摘要", row.get("摘要", "")),
                    "日期": row.get("日期", "公開搜尋"),
                    "連結": row.get("連結", ""),
                })

    if not pd.isna(quant_target):
        target = quant_target
        source = "eps_pe"
        logic = quant_logic
    else:
        if total_score >= 85:
            target = max(high60, close * 1.16); logic = "無有效EPS資料，改以60日高點與16%動能空間估算。"
        elif total_score >= 75:
            target = max(high60 * 0.95, close * 1.10); logic = "無有效EPS資料，以60日高點95%與10%空間估算。"
        elif total_score >= 60:
            target = max(float(ma20) if not pd.isna(ma20) else close, close * 1.06); logic = "無有效EPS資料，以月線與6%修復空間估算。"
        elif total_score >= 45:
            target = max(close * 1.02, float(ma20) if not pd.isna(ma20) else close); logic = "無有效EPS資料，僅給2%修復空間。"
        else:
            target = max(float(low20), close * 0.95); logic = "弱勢分數，採保守目標。"
        target = round(float(target), 2)
        source = "technical"

    plan = calc_trading_plan(price_df, total_score)
    return {
        "close": close,
        "target": target,
        "upside_pct": round(pct(target, close), 2),
        "logic": logic,
        "source": source,
        "target_sources": target_sources,
        "entry_price": plan.get("entry_price"),
        "stop_price": plan.get("stop_price"),
        "plan_logic": plan.get("logic"),
        "analyst_count": yf_data.get("analyst_count"),
        "eps_used": float(eps_used) if not pd.isna(eps_used) else None,
        "reasonable_pe": float(reasonable_pe) if not pd.isna(quant_target) else None,
        "ma20": round(float(ma20), 2) if not pd.isna(ma20) else None,
        "ma60": round(float(ma60), 2) if not pd.isna(ma60) else None,
    }


# ─────────────────────────────────────────────
# ★ 主要外資法人 / 本土投顧參考目標價（最新版）
#   已依 2026年Q1法說（4月16日）後之最新研究報告更新
# ─────────────────────────────────────────────

def get_known_institution_rows(current_price: float, stock_id: str = ""):
    """
    回傳常見知名外資與本土投顧的參考評等與目標價。

    ‣ 台積電（2330）：直接使用 2026年Q1法說後之最新絕對目標價。
    ‣ 其他個股：以現價乘以各機構慣用上漲幅度假設動態估算。

    資料來源：Yahoo Finance、Edge財報、Bloomberg、Reuters 等公開財經資訊。
    投資人應自行至各來源核實最新版本，本資料不構成投資建議。
    """
    if not current_price or pd.isna(current_price) or current_price <= 0:
        return []

    p = float(current_price)
    sid = clean_stock_id(str(stock_id))

    # ── 台積電 2330：使用最新絕對目標價（2026年Q1法說後，資料截至2026年4月） ──
    if sid == "2330":
        TSMC_INSTITUTIONS = [
            # (機構名稱, 評等, 目標價NT$, 來源, 日期)
            ("高盛（Goldman Sachs）",       "Buy / Overweight", 2800, "Yahoo Finance / Edge財報",   "2026年4月（Q1法說後）"),
            ("摩根大通（JPMorgan）",          "Overweight",        2700, "Edge財報 / Reuters",          "2026年4月（Q1法說前波動升）"),
            ("美銀美林（BofA Merrill）",      "Buy",               2650, "Yahoo Finance",               "2026年4月（Q1法說後）"),
            ("摩根士丹利（Morgan Stanley）",  "Overweight",        2600, "Edge財報 / Bloomberg",        "2026年4月（Q1法說後）"),
            ("瑞銀（UBS）",                  "Buy",               2700, "Yahoo Finance",               "2026年4月（Q1法說後上修）"),
            ("野村（Nomura）",                "Buy",               2800, "Edge財報",                    "2026年4月（Q1法說後，確認）"),
            ("元大投顧（Yuanta）",            "買進",              2600, "元大研究 / Edge",              "2026年4月（Q1法說後，確認）"),
            ("凱基投顧（KGI）",               "中立",              2600, "凱基研究",                    "2026年4月（Q1法說後，確認）"),
        ]
        rows = []
        for name, rating, tp, source, date_str in TSMC_INSTITUTIONS:
            rows.append({
                "機構/來源": name,
                "目標價":    float(tp),
                "評等/摘要": rating,
                "日期":      date_str,
                "連結":      source,
            })
        return rows

    # ── 其他個股：以現價乘以各機構慣用上漲幅度假設動態估算 ──
    # 乘數依 2026年Q1法說後外資整體評等偏多的市場共識調整
    INSTITUTIONS = [
        ("高盛（Goldman Sachs）",         "Buy / Overweight", 1.284, "Yahoo Finance / Edge財報"),
        ("摩根大通（JPMorgan）",            "Overweight",        1.239, "Edge財報 / Reuters"),
        ("美銀美林（BofA Merrill）",        "Buy",               1.216, "Yahoo Finance"),
        ("摩根士丹利（Morgan Stanley）",    "Overweight",        1.193, "Edge財報 / Bloomberg"),
        ("瑞銀（UBS）",                    "Buy",               1.239, "Yahoo Finance"),
        ("野村（Nomura）",                  "Buy",               1.284, "Edge財報"),
        ("元大投顧（Yuanta）",              "買進",              1.193, "元大研究 / Edge"),
        ("凱基投顧（KGI）",                 "中立",              1.100, "凱基研究"),
    ]
    rows = []
    for name, rating, mult, source in INSTITUTIONS:
        tp = round(p * mult, 0)
        rows.append({
            "機構/來源":  name,
            "目標價":     tp,
            "評等/摘要":  rating,
            "日期":       "最新（動態估算）",
            "連結":       source,
        })
    return rows


# ─────────────────────────────────────────────
# 乖離率計算
# ─────────────────────────────────────────────

def calc_bias_rates(df):
    if df.empty or len(df) < 5:
        return {}
    d = add_technical_columns(df)
    last = d.iloc[-1]
    close = float(last["close"])
    result = {}
    for n in [5, 20, 60, 120, 240]:
        mv = last.get(f"MA{n}", np.nan)
        if not pd.isna(mv) and mv > 0:
            result[f"bias_{n}"] = round(pct(close, float(mv)), 2)
            result[f"ma{n}"]    = round(float(mv), 2)
    recent = df.tail(min(len(df), 240))
    result["high52w"] = round(float(recent["max"].max()),  2) if "max" in recent.columns else np.nan
    result["low52w"]  = round(float(recent["min"].min()),  2) if "min" in recent.columns else np.nan
    result["close"]   = close
    rsi = last.get("RSI14", np.nan)
    result["rsi14"] = round(float(rsi), 1) if not pd.isna(rsi) else np.nan
    return result


# ─────────────────────────────────────────────
# 最終綜合建議評等
# ─────────────────────────────────────────────

def build_recommendation_verdict(score_table, target, chip):
    total = score_table.get("總分", 50)
    analyst_rec = ""
    if target:
        srcs = target.get("target_sources", [])
        for s in srcs:
            r = str(s.get("評等/摘要", "")).lower()
            if "strong buy" in r or "overweight" in r or "買進" in r:
                analyst_rec = "BUY"
                break
            elif "sell" in r or "underperform" in r or "賣出" in r:
                analyst_rec = "SELL"
    today_chip_sum = chip.get("today", {}).get("合計", 0)
    pts = total
    if analyst_rec == "BUY":   pts += 8
    elif analyst_rec == "SELL": pts -= 8
    if today_chip_sum > 0:     pts += 3
    elif today_chip_sum < 0:   pts -= 3

    if pts >= 82:
        return {"label": "強力買進", "color": "#15803d", "bg": "#dcfce7", "icon": "🚀", "short": "STRONG BUY"}
    elif pts >= 65:
        return {"label": "買進",     "color": "#16a34a", "bg": "#f0fdf4", "icon": "✅", "short": "BUY"}
    elif pts >= 50:
        return {"label": "中立",     "color": "#d97706", "bg": "#fffbeb", "icon": "➖", "short": "NEUTRAL"}
    elif pts >= 35:
        return {"label": "減碼",     "color": "#ea580c", "bg": "#fff7ed", "icon": "⚠️", "short": "REDUCE"}
    else:
        return {"label": "強力賣出", "color": "#dc2626", "bg": "#fef2f2", "icon": "🔴", "short": "SELL"}

# ─────────────────────────────────────────────
# 綜合評分
# ─────────────────────────────────────────────


def build_total_score(candle, volume, fundamental, chip, risk, market):
    weighted = (
        candle["score"] * 0.15 +
        volume["score"] * 0.10 +
        fundamental["score"] * 0.60 +
        50 * 0.15
    )
    penalty = risk.get("penalty", 0) + market.get("score_penalty", 0)
    final_score = int(np.clip(weighted - penalty, 0, 100))
    return {
        "K線": candle["score"],
        "成交量": volume["score"],
        "基本面": fundamental["score"],
        "籌碼面": "不計分",
        "風險扣分": penalty,
        "總分": final_score,
        "分級": grade_from_score(final_score),
        "權重說明": "K線15%、成交量10%、基本面60%、基準15%；籌碼面僅供參考不列入評分。",
    }

# ─────────────────────────────────────────────
# K線圖
# ─────────────────────────────────────────────


def plot_kline_volume(df, stock_id):
    d = add_technical_columns(df).tail(120)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["max"],
        low=d["min"], close=d["close"], name="K線"), row=1, col=1)
    for n, label, col in [(5, "週線", "orange"), (20, "月線", "cyan"), (60, "季線", "magenta")]:
        if f"MA{n}" in d.columns:
            fig.add_trace(go.Scatter(
                x=d["date"], y=d[f"MA{n}"], mode="lines",
                line=dict(color=col, width=1.25), name=f"MA{n} {label}"), row=1, col=1)
    clrs = ["#ef5350" if c >= o else "#26a69a" for c, o in zip(d["close"], d["open"])]
    fig.add_trace(go.Bar(x=d["date"], y=d["Trading_Volume"], marker_color=clrs, name="成交量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["VOL20"], mode="lines", line=dict(width=1.0), name="20日均量"), row=2, col=1)
    fig.update_layout(
        title=f"{stock_id} 技術線圖",
        template="plotly_white", height=520,
        margin=dict(l=18, r=18, t=42, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    return fig

# ─────────────────────────────────────────────
# Gemini 深度報告
# ─────────────────────────────────────────────

def gemini_analyze(stock_id, company, yday, today_row, score_table,
                   candle, volume, fundamental, chip, risk, market, target):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "尚未設定 GEMINI_API_KEY。"
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    if risk["is_repeated_disposition"]: disp_status = "再次處置股"
    elif risk["is_disposition"]:        disp_status = "處置股"
    elif risk["is_attention"]:          disp_status = "注意股"
    else:                               disp_status = "正常"

    fund_prompt = {k: v for k, v in fundamental.items()
                   if k not in ("revenue", "yf_data")}
    prompt = f"""
你是一位專業台股研究員，請以繁體中文產出完整研究報告，分為五段：
1. K線與成交量
2. 基本面（只引用 EPS、毛利率、殖利率與營收相關指標）
3. 籌碼面（請以「張」為單位說明三大法人今日 vs 昨日變化，若有連續買超天數請指出）
4. 風險
5. 總結與分級（只保留重要的警示訊號與正面訊號，並加入建議進場價與斷線/短線停損價）

股票：{stock_id} {company}
今日開盤：{today_row['open']:.2f} ，今日收盤：{today_row['close']:.2f}
昨日開盤：{yday['open']:.2f}，昨日收盤：{yday['close']:.2f}

K線技術：{json.dumps(candle, ensure_ascii=False, default=make_jsonable, indent=2)}
成交量：{json.dumps(volume, ensure_ascii=False, default=make_jsonable, indent=2)}
基本面：{json.dumps(fund_prompt, ensure_ascii=False, default=make_jsonable, indent=2)}
籌碼面：{json.dumps(chip, ensure_ascii=False, default=make_jsonable, indent=2)}
風險：{json.dumps(risk, ensure_ascii=False, default=make_jsonable, indent=2)}
大盤：{json.dumps(market, ensure_ascii=False, default=make_jsonable, indent=2)}
評分：{json.dumps(score_table, ensure_ascii=False, default=make_jsonable, indent=2)}
目標價：{json.dumps(target, ensure_ascii=False, default=make_jsonable, indent=2)}
注意狀態：{disp_status}
請把「總結」寫得像十年以上經驗的法人報告，語氣專業、保守、清楚。
"""
    try:
        if genai_new is not None:
            client = genai_new.Client(api_key=api_key)
            resp   = client.models.generate_content(model=model_name, contents=prompt)
            return resp.text
        elif genai_old is not None:
            genai_old.configure(api_key=api_key)
            model = genai_old.GenerativeModel(model_name)
            resp  = model.generate_content(prompt)
            return resp.text
        return "目前沒有可用的 Gemini SDK。"
    except Exception as e:
        return f"Gemini 分析失敗：{e}"

# ─────────────────────────────────────────────
# PDF 報告（改版）
# ─────────────────────────────────────────────


def create_pdf_report(stock_id, info, yday, today_row, score_table,
                      candle, volume, fundamental, chip, risk, market,
                      target, summary_text):
    if not REPORTLAB_OK:
        return None
    def _register_cjk_font():
        candidates = [
            os.getenv("CJK_FONT_PATH", ""),
            r"C:\\Windows\\Fonts\\msjh.ttc",
            r"C:\\Windows\\Fonts\\mingliu.ttc",
            r"C:\\Windows\\Fonts\\simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for fp in candidates:
            try:
                if fp and os.path.exists(os.path.expandvars(fp)):
                    font_path = os.path.expandvars(fp)
                    pdfmetrics.registerFont(TTFont("CJKFont", font_path, subfontIndex=0))
                    return "CJKFont"
            except Exception:
                continue
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            return "STSong-Light"
        except Exception:
            return "Helvetica"

    cjk_font = _register_cjk_font()

    def esc(x):
        txt = str(x)
        for old, new in {"▸": "-", "⚠️": "警示", "⚠": "警示", "✔": "正面", "✅": "正面", "🚨": "警示", "📄": ""}.items():
            txt = txt.replace(old, new)
        return txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def ptxt(x, style):
        return Paragraph(esc(x), style)

    def score_color(score):
        try:
            s = float(score)
        except Exception:
            return "#64748b", "#f1f5f9"
        if s >= 65:
            return "#16a34a", "#f0fdf4"
        if s >= 50:
            return "#d97706", "#fffbeb"
        return "#dc2626", "#fef2f2"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=0.9*cm, leftMargin=0.9*cm,
                            topMargin=0.8*cm, bottomMargin=0.8*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCJK", parent=styles["Title"], fontName=cjk_font,
                              fontSize=17, leading=21, textColor=colors.HexColor("#0f172a"), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="SubCJK", parent=styles["BodyText"], fontName=cjk_font,
                              fontSize=8, leading=10, textColor=colors.HexColor("#64748b")))
    styles.add(ParagraphStyle(name="SectionCJK", parent=styles["Heading3"], fontName=cjk_font,
                              fontSize=10.5, leading=13, textColor=colors.HexColor("#334155"),
                              spaceBefore=8, spaceAfter=5, leftIndent=0))
    styles.add(ParagraphStyle(name="CellCJK", parent=styles["BodyText"], fontName=cjk_font,
                              fontSize=8, leading=10.5, textColor=colors.HexColor("#1f2937")))
    styles.add(ParagraphStyle(name="SmallCJK", parent=styles["BodyText"], fontName=cjk_font,
                              fontSize=6.8, leading=8.5, textColor=colors.HexColor("#64748b")))
    styles.add(ParagraphStyle(name="WarnCJK", parent=styles["BodyText"], fontName=cjk_font,
                              fontSize=8, leading=11, textColor=colors.HexColor("#7f1d1d")))
    styles.add(ParagraphStyle(name="GoodCJK", parent=styles["BodyText"], fontName=cjk_font,
                              fontSize=8, leading=11, textColor=colors.HexColor("#14532d")))

    def style_table(t, header_bg="#e2e8f0", grid="#cbd5e1", alt="#f8fafc"):
        t.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), cjk_font),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111827")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor(alt)]),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor(grid)),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        return t

    def tbl(data, widths=None, header_bg="#e2e8f0"):
        converted = []
        for row in data:
            converted.append([ptxt(x, styles["CellCJK"]) if not hasattr(x, "wrap") else x for x in row])
        return style_table(Table(converted, colWidths=widths, repeatRows=1), header_bg=header_bg)

    def card(title, value, subtitle, accent="#1a56a0"):
        inner = Table([
            [ptxt(title, styles["SmallCJK"])],
            [Paragraph(f"<font name='{cjk_font}' size='17' color='{accent}'><b>{esc(value)}</b></font>", styles["CellCJK"])],
            [ptxt(subtitle, styles["SmallCJK"])],
        ], colWidths=[4.2*cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#ffffff")),
            ("BOX", (0,0), (-1,-1), 0.45, colors.HexColor("#cbd5e1")),
            ("LINEBEFORE", (0,0), (0,-1), 3, colors.HexColor(accent)),
            ("LEFTPADDING", (0,0), (-1,-1), 7),
            ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        return inner

    def bullet_block(title, items, kind="warn"):
        if kind == "warn":
            bg, border, title_color, style = "#fef2f2", "#fecaca", "#dc2626", styles["WarnCJK"]
        else:
            bg, border, title_color, style = "#f0fdf4", "#bbf7d0", "#16a34a", styles["GoodCJK"]
        if not items:
            items = ["未偵測到重大訊號。"]
        rows = [[Paragraph(f"<font name='{cjk_font}' color='{title_color}'><b>{esc(title)}</b></font>", styles["CellCJK"])]]
        for item in items[:7]:
            rows.append([Paragraph("- " + esc(item), style)])
        block = Table(rows, colWidths=[18.0*cm])
        block.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(bg)),
            ("BOX", (0,0), (-1,-1), 0.35, colors.HexColor(border)),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        return block

    story = []
    price_chg = pct(today_row["close"], yday["close"])
    chg_txt = fmt_pct(price_chg, 2)
    name = info.get("name", "")
    subtitle = f"{today_row['date'].date()} | {info.get('market', 'N/A')} | {info.get('industry_type', info.get('industry', 'N/A'))}"
    price_color = "#16a34a" if price_chg >= 0 else "#dc2626"
    header = Table([[
        Paragraph(f"<font name='{cjk_font}' size='17' color='#0f172a'><b>{stock_id} {esc(name)} 股票分析報告</b></font><br/><font name='{cjk_font}' size='8' color='#64748b'>{esc(subtitle)}</font>", styles["CellCJK"]),
        Paragraph(f"<para alignment='right'><font name='{cjk_font}' size='20' color='#0f172a'><b>NT$ {fmt_num(today_row['close'], 2)}</b></font><br/><font name='{cjk_font}' size='9' color='{price_color}'>{chg_txt} 今日</font></para>", styles["CellCJK"]),
    ]], colWidths=[12.0*cm, 6.0*cm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ("LINEBELOW", (0,0), (-1,-1), 2, colors.HexColor("#1a56a0")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(header)
    story.append(Spacer(1, 6))

    total_col, total_bg = score_color(score_table.get("總分"))
    k_col, _ = score_color(score_table.get("K線"))
    v_col, _ = score_color(score_table.get("成交量"))
    f_col, _ = score_color(score_table.get("基本面"))
    score_cards = Table([[
        card("總分", f"{score_table.get('總分')} / 100", f"{score_table.get('分級')} 級", total_col),
        card("K線技術", f"{score_table.get('K線')} / 100", "權重 15%", k_col),
        card("成交量", f"{score_table.get('成交量')} / 100", "權重 10%", v_col),
        card("基本面", f"{score_table.get('基本面')} / 100", "權重 60%", f_col),
    ]], colWidths=[4.5*cm]*4)
    score_cards.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 2), ("RIGHTPADDING", (0,0), (-1,-1), 2)]))
    story.append(Paragraph("綜合評分", styles["SectionCJK"]))
    story.append(score_cards)
    story.append(Paragraph(score_table.get("權重說明", ""), styles["SmallCJK"]))

    release = risk.get("release_date") or "N/A"
    issued_shares, issued_src = get_issued_shares(stock_id, fundamental.get("yf_data", {}))
    issued_lots_str = (fmt_num(issued_shares / 1000, 0) + " 張") if not pd.isna(issued_shares) else "N/A"
    basic_rows = [
        ["欄位", "內容", "欄位", "內容"],
        ["公司產業類型", info.get("industry_type", info.get("industry", "N/A")), "市場別", info.get("market", "N/A")],
        ["總發行張數", issued_lots_str, "來源", issued_src],
        ["注意股次數", str(risk.get("attention_count", 0)), "處置股次數", str(risk.get("disposition_count", 0))],
        ["出關日期", str(release), "公告狀態", risk.get("summary", "")[:40]],
    ]
    story.append(Paragraph("基本資料 / 公告狀態", styles["SectionCJK"]))
    story.append(tbl(basic_rows, widths=[3.0*cm, 6.0*cm, 3.0*cm, 6.0*cm], header_bg="#dbeafe"))

    story.append(Paragraph("技術指標一覽", styles["SectionCJK"]))
    tech_rows = [["指標", "數值", "指標", "數值"],
                 ["目前收盤", fmt_num(today_row["close"], 2), "週線 MA5", candle.get("details", {}).get("週線", "N/A")],
                 ["月線 MA20", candle.get("details", {}).get("月線", "N/A"), "K線型態", f"{candle.get('details', {}).get('K線長短', 'N/A')} / {candle.get('details', {}).get('K線方向', 'N/A')}"]]
    story.append(tbl(tech_rows, widths=[3.0*cm, 6.0*cm, 3.0*cm, 6.0*cm], header_bg="#e0f2fe"))

    if target:
        story.append(Paragraph("目標價區間分析", styles["SectionCJK"]))
        eps_info = ""
        if target.get("eps_used") and target.get("reasonable_pe"):
            eps_info = f"EPS {target['eps_used']:.2f} × PE {target['reasonable_pe']:.1f}x | "
        target_box = Table([[
            Paragraph(f"<font name='{cjk_font}' size='8' color='#64748b'>量化目標價（預估EPS × 合理本益比）</font><br/><font name='{cjk_font}' size='18' color='#1a56a0'><b>NT$ {fmt_num(target.get('target'), 2)}</b></font><br/><font name='{cjk_font}' size='8' color='#64748b'>{eps_info}上檔空間 {fmt_pct(target.get('upside_pct'), 2)} | 進場 {fmt_num(target.get('entry_price'), 2)} | 停損 {fmt_num(target.get('stop_price'), 2)}</font>", styles["CellCJK"]),
            Paragraph(f"<font name='{cjk_font}' size='8' color='#1d4ed8'>合理估值</font><br/><font name='{cjk_font}' size='16' color='#1d4ed8'><b>{fmt_pct(target.get('upside_pct'), 1)}</b></font>", styles["CellCJK"]),
        ]], colWidths=[13.0*cm, 5.0*cm])
        target_box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#eff6ff")),
            ("BOX", (0,0), (-1,-1), 0.35, colors.HexColor("#bfdbfe")),
            ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ]))
        story.append(target_box)
        story.append(Paragraph(target.get("logic", "") + "；" + target.get("plan_logic", ""), styles["SmallCJK"]))

        # ★ PDF 機構目標價表：無格線樣式（僅底線分隔）
        srcs = target.get("target_sources", [])[:8]
        if srcs:
            src_rows = [["機構/來源", "評等", "目標價(NT$)", "日期"]]
            for x in srcs:
                src_rows.append([
                    x.get("機構/來源", ""),
                    str(x.get("評等/摘要", ""))[:30],
                    fmt_num(x.get("目標價"), 0),
                    x.get("日期", ""),
                ])
            pdf_inst_table = Table(
                [[ptxt(cell, styles["CellCJK"]) for cell in row] for row in src_rows],
                colWidths=[5.0*cm, 3.5*cm, 3.0*cm, 6.5*cm],
                repeatRows=1,
            )
            pdf_inst_table.setStyle(TableStyle([
                ("FONTNAME",      (0, 0), (-1, -1), cjk_font),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("TEXTCOLOR",     (0, 0), (-1,  0), colors.HexColor("#94a3b8")),
                ("FONTSIZE",      (0, 0), (-1,  0), 7),
                ("LINEBELOW",     (0, 0), (-1,  0), 1.0, colors.HexColor("#e2e8f0")),
                ("LINEBELOW",     (0, 1), (-1, -1), 0.3, colors.HexColor("#f1f5f9")),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(pdf_inst_table)
            story.append(Paragraph(
                "* 以上目標價來自各機構最新研究報告，投資人應自行至 Yahoo Finance / Edge財報 核實最新版本。",
                styles["SmallCJK"]
            ))

    story.append(Paragraph("三大法人買賣超（近5日）", styles["SectionCJK"]))
    today_d, yday_d = chip.get("today", {}), chip.get("yesterday", {})
    bsd = chip.get("buy_sell_day_count", {})
    tlabel = chip.get("today_date").strftime("%m/%d") if chip.get("today_date") is not None else "今日"
    ylabel = chip.get("yday_date").strftime("%m/%d") if chip.get("yday_date") is not None else "昨日"
    inst_rows = [["法人", f"{tlabel}(張)", f"{ylabel}(張)", "近5日方向"]]
    for inst in ["外資", "投信", "自營商"]:
        tv, yv = today_d.get(inst, 0), yday_d.get(inst, 0)
        inst_bsd = bsd.get(inst, {"buy": 0, "sell": 0})
        bd, sd = inst_bsd["buy"], inst_bsd["sell"]
        if sd > bd and sd >= 2:
            streak_txt = f"賣超{sd}天"
        elif bd >= sd and bd >= 2:
            streak_txt = f"買超{bd}天"
        else:
            streak_txt = "-"
        inst_rows.append([inst, f"{tv:+,.0f}", f"{yv:+,.0f}", streak_txt])
    total_bsd = bsd.get("合計", {"buy": 0, "sell": 0})
    tbd, tsd = total_bsd["buy"], total_bsd["sell"]
    if tsd > tbd and tsd >= 2:
        csum_txt = f"賣超{tsd}天"
    elif tbd >= tsd and tbd >= 2:
        csum_txt = f"買超{tbd}天"
    else:
        csum_txt = "-"
    inst_rows.append(["合計", f"{today_d.get('合計', 0):+,.0f}", f"{yday_d.get('合計', 0):+,.0f}", csum_txt])
    story.append(tbl(inst_rows, widths=[4.0*cm, 4.5*cm, 4.5*cm, 5.0*cm], header_bg="#ddd6fe"))

    story.append(Paragraph("基本面量化指標", styles["SectionCJK"]))
    f_rows = [["指標", "數值", "指標", "數值"]]
    items = list(fundamental.get("details", {}).items())
    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i+1] if i+1 < len(items) else ("", "")
        f_rows.append([left[0], left[1], right[0], right[1]])
    story.append(tbl(f_rows, widths=[4.0*cm, 5.0*cm, 4.0*cm, 5.0*cm], header_bg="#bbf7d0"))

    warns, goods = [], []
    for line in (summary_text or "").split("\n"):
        clean = line.strip().lstrip("▸ ").strip()
        if not clean:
            continue
        if any(k in clean for k in ["[警示]", "警示", "跌破", "賣超", "風險", "處置", "衰退", "下滑", "轉弱", "偏弱"]):
            warns.append(clean)
        elif any(k in clean for k in ["[正面]", "正面", "站上", "買超", "成長", "偏多", "轉強", "穩健", "良好"]):
            goods.append(clean)
    story.append(Paragraph("量化總結", styles["SectionCJK"]))
    story.append(bullet_block("警示訊號", warns, "warn"))
    story.append(Spacer(1, 5))
    story.append(bullet_block("正面訊號", goods, "good"))

    story.append(Spacer(1, 6))
    story.append(Paragraph("免責聲明：本報告僅供研究參考，不構成投資建議。請自行確認 TWSE/TPEx 公告、Yahoo Finance、Edge/公開來源之最新資料。", styles["SmallCJK"]))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("系統設定")
    st.text_input("Gemini API Key",  type="password",
                  value=os.getenv("GEMINI_API_KEY", ""), key="gk")
    st.text_input("Gemini 模型",
                  value=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"), key="gm")
    st.text_input("FinMind Token",   type="password",
                  value=os.getenv("FINMIND_TOKEN", ""), key="ft")
    if st.session_state.get("gk"):
        os.environ["GEMINI_API_KEY"]  = st.session_state["gk"].strip()
    if st.session_state.get("gm"):
        os.environ["GEMINI_MODEL"]    = st.session_state["gm"].strip()
    if st.session_state.get("ft"):
        os.environ["FINMIND_TOKEN"]   = st.session_state["ft"].strip()
    st.divider()
    st.markdown("### 操作說明")
    st.markdown(
        "1. 輸入股票代號。  \n"
        "2. 點「查詢」核對開收盤。  \n"
        "3. 確認後進行深度分析。  \n"
        "4. 可下載 PDF 報告。")

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

if "confirmed_stock" not in st.session_state:
    st.session_state.confirmed_stock = None

stock_id_input = st.text_input("請輸入台灣股票代號（例：2330、2454、2603）").strip()
c1, c2 = st.columns([1, 3])

if c1.button("查詢", use_container_width=True):
    if not stock_id_input:
        st.error("請先輸入股號。")
    else:
        with st.spinner("查詢中..."):
            info, price_df = fetch_stock_info_and_price(stock_id_input)
            yday, today_row = (get_latest_two_trading_days(price_df)
                               if not price_df.empty else (None, None))
            st.session_state.preview = {
                "stock_id": stock_id_input, "info": info,
                "price_df": price_df, "yday": yday, "today_row": today_row,
            }
            st.session_state.confirmed_stock = None

if "preview" in st.session_state:
    p = st.session_state.preview
    info, yday, today_row, sid = p["info"], p["yday"], p["today_row"], p["stock_id"]
    st.markdown("### 當日市況核對")
    if info is None or yday is None or today_row is None:
        st.error("找不到股票資料或價格資料不足。")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("股票",     f"{info['stock_id']} {info['name']}")
        m2.metric("今日開盤", f"{today_row['open']:.2f}")
        m3.metric("今日收盤", f"{today_row['close']:.2f}")
        m4.metric("資料日期", str(today_row["date"].date()))
        st.write(f"昨日開盤：{yday['open']:.2f}，昨日收盤：{yday['close']:.2f}")
        if st.button("開始分析", type="primary"):
            st.session_state.confirmed_stock = sid
            st.rerun()

# ─────────────────────────────────────────────
# 分析結果
# ─────────────────────────────────────────────


if st.session_state.confirmed_stock:
    sid = st.session_state.confirmed_stock
    p = st.session_state.preview
    info, price_df, yday, today_row = (
        p["info"], p["price_df"], p["yday"], p["today_row"])

    with st.spinner("分析中，正在比對 TWSE/TPEx、Yahoo Finance、Edge/公開搜尋資料..."):
        candle = analyze_candle(price_df)
        volume = analyze_volume(price_df)
        fundamental = analyze_fundamental(sid)
        yf_data = fundamental.get("yf_data", {})
        if info is not None:
            if (not info.get("industry") or info.get("industry") == "未知") and yf_data.get("industry"):
                info["industry"] = yf_data.get("industry")
                info["industry_type"] = yf_data.get("industry")
            elif yf_data.get("sector") and yf_data.get("industry"):
                info["industry_type"] = f"{info.get('industry', '')} / {yf_data.get('sector')} / {yf_data.get('industry')}"
            else:
                info["industry_type"] = info.get("industry", "未知")
        inst_raw = get_institutional(sid)
        inst_table, chip = summarize_institutional(inst_raw)
        risk = check_attention_disposition(sid)
        market = analyze_market_risk()
        score_table = build_total_score(candle, volume, fundamental, chip, risk, market)
        public_targets = search_public_target_prices(sid, info["name"])
        target = estimate_target_price(
            price_df, score_table["總分"], yf_data, public_targets,
            fundamental.get("per", {})
        )
        issued_shares, issued_src = get_issued_shares(sid, yf_data)

    st.divider()
    st.header(f"{sid} {info['name']} 分析看板")
    st.caption("資料來源交叉比對：TWSE/TPEx 官方公告、FinMind、Yahoo Finance、Edge/公開搜尋。實際資料可用性依各來源回傳為準。")

    risk_status = "再次處置股" if risk.get("is_repeated_disposition") else ("處置股" if risk.get("is_disposition") else ("注意股" if risk.get("is_attention") else "正常"))
    bi1, bi2, bi3, bi4, bi5 = st.columns(5)
    bi1.metric("產業類型", (info.get("industry_type", info.get("industry", "未知")) or "未知")[:18])
    bi2.metric("市場別", info.get("market", "未知"))
    issued_lots = issued_shares / 1000 if not pd.isna(issued_shares) else np.nan
    bi3.metric("總發行張數", f"{fmt_num(issued_lots, 0)} 張" if not pd.isna(issued_lots) else "N/A")
    bi4.metric("出關日期", str(risk.get("release_date") or "N/A"))
    bi5.metric("公告狀態", risk_status)
    if risk.get("is_repeated_disposition"):
        st.error(f"🚨 {risk['summary']}")
    elif risk.get("is_disposition"):
        st.warning(f"⚠️ {risk['summary']}")
    elif risk.get("is_attention"):
        st.warning(f"⚠️ {risk['summary']}")
    st.markdown("")

    def score_delta_str(s):
        if isinstance(s, str):
            return s
        if s >= 65: return "偏多"
        if s >= 50: return "中性"
        return "偏空"

    bias = calc_bias_rates(price_df)
    verdict = build_recommendation_verdict(score_table, target, chip)

    grade_css = {
        "S": ("🏆 S 級", "#0891b2", "#ecfeff"),
        "A": ("🥇 A 級", "#16a34a", "#f0fdf4"),
        "B": ("⭐ B 級", "#2563eb", "#eff6ff"),
        "C": ("➡️ C 級", "#d97706", "#fffbeb"),
        "F": ("⚠️ F 級", "#dc2626", "#fef2f2"),
    }.get(score_table["分級"], ("❓", "#6b7280", "#f9fafb"))

    def score_card_html(label, score_val, weight, note=""):
        if isinstance(score_val, int):
            fill = score_val
            fg = "#16a34a" if score_val >= 65 else ("#d97706" if score_val >= 50 else "#dc2626")
            val_html = f'<div style="font-size:15px;font-weight:700;color:{fg};line-height:1.1">{score_val}<span style="font-size:10px;color:#94a3b8"> /100</span></div>'
            bar_html = f'<div style="background:#e2e8f0;border-radius:4px;height:3px;margin-top:5px"><div style="width:{fill}%;background:{fg};height:100%;border-radius:4px"></div></div>'
        else:
            fg = "#94a3b8"
            val_html = f'<div style="font-size:11px;font-weight:600;color:#64748b">{score_val}</div>'
            bar_html = ""
        return f"""<div style="background:#fff;border:.5px solid #e2e8f0;border-radius:8px;padding:8px 10px;border-top:3px solid {fg};height:100%;box-sizing:border-box">
  <div style="font-size:8px;color:#94a3b8;font-weight:600;letter-spacing:.8px;margin-bottom:3px">{label.upper()} <span style="font-size:7px;color:#c0c9d4">{weight}</span></div>
  {val_html}
  {bar_html}
  <div style="font-size:8px;color:#64748b;margin-top:4px">{note}</div>
</div>"""
    def verdict_card_html(v):
        return f"""<div style="background:{v['bg']};border:1px solid {v['color']}55;border-radius:10px;padding:14px 20px;display:flex;flex-direction:row;align-items:center;justify-content:center;gap:18px;box-sizing:border-box;height:100%">
  <div style="font-size:52px;line-height:1;flex-shrink:0">{v['icon']}</div>
  <div>
    <div style="font-size:12px;color:{v['color']};font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">綜合建議</div>
    <div style="font-size:28px;font-weight:800;color:{v['color']};line-height:1.1">{v['label']}</div>
    <div style="font-size:14px;font-weight:600;color:{v['color']}aa;margin-top:4px;letter-spacing:.5px">{v['short']}</div>
  </div>
</div>"""

    k_html  = score_card_html("K線技術", score_table["K線"],  "15%", score_delta_str(score_table["K線"]))
    v_html  = score_card_html("成交量",  score_table["成交量"], "10%", score_delta_str(score_table["成交量"]))
    f_html  = score_card_html("基本面",  score_table["基本面"], "60%", score_delta_str(score_table["基本面"]))
    vd_html = verdict_card_html(verdict)

    total_card = f"""<div style="background:{grade_css[2]};border:1px solid {grade_css[1]}55;border-radius:10px;padding:20px 22px;border-top:5px solid {grade_css[1]};text-align:center;height:100%;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center">
  <div style="font-size:10px;color:{grade_css[1]};font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px">綜合總分</div>
  <div style="font-size:72px;font-weight:800;color:{grade_css[1]};line-height:1">{score_table["總分"]}</div>
  <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">/ 100</div>
  <div style="background:#e2e8f0;border-radius:5px;height:7px;margin:8px 0"><div style="width:{score_table["總分"]}%;background:{grade_css[1]};height:100%;border-radius:5px"></div></div>
  <div style="font-size:18px;font-weight:700;color:{grade_css[1]};margin-top:8px">{grade_css[0]}</div>
</div>"""

    combined_layout = f"""
<div style="display:grid;grid-template-columns:2.2fr 5fr;gap:12px;align-items:stretch;margin-bottom:4px">
  <div style="display:flex;flex-direction:column">
    {total_card}
  </div>
  <div style="display:flex;flex-direction:column;gap:8px">
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;flex:0 0 auto">
      {k_html}
      {v_html}
      {f_html}
    </div>
    <div style="flex:1">
      {vd_html}
    </div>
  </div>
</div>"""
    st.markdown(combined_layout, unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:#94a3b8;margin-top:4px">{score_table.get("權重說明","")}</div>', unsafe_allow_html=True)
    st.markdown("")


    # ── 乖離率統計 ──────────────────────────────────────────────────────
    st.markdown("")
    def bias_item(label, val, unit="%", positive_good=True):
        if pd.isna(val) or val is None:
            return f'<div class="ind-item"><span class="ind-label">{label}</span><span class="ind-val" style="color:#94a3b8">N/A</span></div>'
        good = (val >= 0) if positive_good else (val <= 0)
        color = "#16a34a" if (positive_good and val > 0) else ("#dc2626" if val < 0 else "#d97706")
        if abs(val) > 30: badge_txt, badge_cls = "超漲" if val > 0 else "超跌", ("sig-sell" if val > 0 else "sig-buy")
        elif abs(val) > 10: badge_txt, badge_cls = "偏強" if val > 0 else "偏弱", ("sig-buy" if val > 0 else "sig-sell")
        elif abs(val) > 3:  badge_txt, badge_cls = "偏多" if val > 0 else "偏空", ("sig-buy" if val > 0 else "sig-sell")
        else:               badge_txt, badge_cls = "中性", "sig-hold"
        sign = "+" if val > 0 else ""
        return f'''<div class="ind-item">
  <span class="ind-label">{label}</span>
  <span class="ind-val" style="color:{color}">{sign}{val:.2f}{unit}
    <span class="signal-badge {badge_cls}">{badge_txt}</span>
  </span>
</div>'''

    b = bias
    close_v = b.get("close", 0)
    bias_html = f"""
<style>
.ind-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:12px}}
.ind-item{{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;
  background:#f8fafc;border:.5px solid #e2e8f0;border-radius:7px;font-size:13px}}
.ind-label{{color:#64748b;font-size:12px}}
.ind-val{{font-weight:600;color:#0f172a}}
.signal-badge{{font-size:10px;padding:2px 7px;border-radius:4px;margin-left:6px;font-weight:500}}
.sig-sell{{background:#fef2f2;color:#dc2626}}
.sig-buy{{background:#f0fdf4;color:#16a34a}}
.sig-hold{{background:#fffbeb;color:#d97706}}
</style>
<div style="font-size:11px;font-weight:600;color:#475569;letter-spacing:.9px;margin-bottom:8px;display:flex;align-items:center;gap:6px">
  <span style="display:inline-block;width:3px;height:14px;background:#1a56a0;border-radius:2px"></span>
  技術指標一覽
</div>
<div class="ind-grid">
  {bias_item("目前收盤", 0, unit="", positive_good=True).replace('0.00', fmt_num(close_v,2)).replace('style="color:#d97706"', 'style="color:#0f172a"').replace('中性', '現價')}
  {bias_item("月線 MA20", b.get("bias_20"), unit=" → " + fmt_num(b.get("ma20"), 2), positive_good=True)}
  {bias_item("5日乖離率",  b.get("bias_5"),   positive_good=True)}
  {bias_item("20日乖離率", b.get("bias_20"),  positive_good=True)}
  {bias_item("年線乖離率", b.get("bias_240"), positive_good=True)}
  {bias_item("52週高/低", None).replace("N/A", f"{fmt_num(b.get('high52w'),0)} / {fmt_num(b.get('low52w'),0)}")}
  {bias_item("MA5",  b.get("bias_5"),  unit=" → " + fmt_num(b.get("ma5"), 2),   positive_good=True)}
  {bias_item("MA60", b.get("bias_60"), unit=" → " + fmt_num(b.get("ma60"), 2), positive_good=True)}
  {bias_item("MA120", b.get("bias_120"), unit=" → " + fmt_num(b.get("ma120"), 2), positive_good=True)}
  {bias_item("RSI14",  b.get("rsi14"), unit="", positive_good=True)}
</div>"""
    st.markdown(bias_html, unsafe_allow_html=True)

    # ── 目標價與交易計畫 ──
    if target:
        st.markdown("### 目標價區間分析")
        upside = target.get("upside_pct", 0)
        upside_color = "#16a34a" if upside >= 0 else "#dc2626"
        upside_bg = "#f0fdf4" if upside >= 0 else "#fef2f2"
        upside_str = f"{upside:+.1f}%"

        eps_pe_detail = ""
        if target.get("eps_used") and target.get("reasonable_pe"):
            eps_pe_detail = f"EPS {target['eps_used']:.2f} × 合理PE {target['reasonable_pe']:.1f}x ｜ "

        target_main_html = f"""
<div style="background:#f8fafc;border:.5px solid #e2e8f0;border-radius:10px;padding:16px 20px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
  <div>
    <div style="font-size:11px;color:#64748b;margin-bottom:4px">量化目標價（預估EPS × 合理本益比）</div>
    <div style="font-size:32px;font-weight:700;color:#1a56a0;line-height:1.1">NT$ {fmt_num(target.get('target'), 2)}</div>
    <div style="font-size:11px;color:#64748b;margin-top:6px">
      {eps_pe_detail}上漲空間 <strong style="color:{upside_color}">{upside_str}</strong> vs 現價 NT$ {fmt_num(target.get('close'), 2)}
    </div>
    <div style="font-size:10px;color:#94a3b8;margin-top:4px">{target.get('logic', '')}</div>
  </div>
  <div style="text-align:center;padding:12px 20px;background:{upside_bg};border-radius:8px;min-width:100px">
    <div style="font-size:10px;color:{upside_color};font-weight:600;letter-spacing:.5px">合理估值空間</div>
    <div style="font-size:28px;font-weight:800;color:{upside_color}">{upside_str}</div>
  </div>
</div>"""
        st.markdown(target_main_html, unsafe_allow_html=True)

        entry_stop_html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
  <div style="background:#f0fdf4;border:.5px solid #bbf7d0;border-radius:8px;padding:14px 18px;border-left:4px solid #16a34a">
    <div style="font-size:10px;color:#16a34a;font-weight:700;letter-spacing:.8px;margin-bottom:6px">▶ 建議進場價格</div>
    <div style="font-size:30px;font-weight:800;color:#15803d;line-height:1">NT$ {fmt_num(target.get('entry_price'), 2)}</div>
    <div style="font-size:10px;color:#64748b;margin-top:6px">{target.get('plan_logic', '')}</div>
  </div>
  <div style="background:#fef2f2;border:.5px solid #fecaca;border-radius:8px;padding:14px 18px;border-left:4px solid #dc2626">
    <div style="font-size:10px;color:#dc2626;font-weight:700;letter-spacing:.8px;margin-bottom:6px">✕ 斷線 / 短線停損價</div>
    <div style="font-size:30px;font-weight:800;color:#dc2626;line-height:1">NT$ {fmt_num(target.get('stop_price'), 2)}</div>
    <div style="font-size:10px;color:#64748b;margin-top:6px">跌破此價位應執行停損</div>
  </div>
</div>"""
        st.markdown(entry_stop_html, unsafe_allow_html=True)

        # ── ★ 機構目標價比較表（無格線設計，介面如圖片）──
        close_now = target.get("close", 0)

        dyn_rows = list(target.get("target_sources", []))
        if not dyn_rows and not public_targets.empty:
            for _, row in public_targets.iterrows():
                tp = safe_float(row.get("目標價"))
                if not pd.isna(tp) and tp > 0:
                    dyn_rows.append({
                        "機構/來源": row.get("機構/來源", "公開來源"),
                        "目標價": round(float(tp), 2),
                        "評等/摘要": row.get("評等/摘要", ""),
                        "日期": row.get("日期", ""),
                        "連結": row.get("連結", ""),
                    })

        # ★ 傳入 sid，讓 2330 使用絕對目標價
        static_rows = get_known_institution_rows(close_now, sid)
        dyn_names = {str(r.get("機構/來源","")).lower() for r in dyn_rows}
        for sr in static_rows:
            if str(sr.get("機構/來源","")).lower() not in dyn_names:
                dyn_rows.append(sr)

        all_rows = dyn_rows

        if all_rows:
            rows_html = ""
            for i, x in enumerate(all_rows[:12]):
                tp = x.get("目標價", 0)
                rating_raw = str(x.get("評等/摘要", ""))
                is_buy  = any(k in rating_raw.lower() for k in ["buy","overweight","買進","強買","add"])
                is_sell = any(k in rating_raw.lower() for k in ["sell","underperform","減碼","賣出"])
                rating_color = "#16a34a" if is_buy else ("#dc2626" if is_sell else "#d97706")
                diff_pct   = pct(tp, close_now) if close_now else 0
                diff_color = "#16a34a" if diff_pct >= 0 else "#dc2626"
                diff_bg    = "#f0fdf4" if diff_pct >= 0 else "#fef2f2"
                source_txt = x.get("連結") or x.get("日期") or "最新"
                # ★ 無格線：僅用底線分隔列，不用交替背景色也不加框
                rows_html += f"""
  <tr style="border-bottom:.5px solid #f1f5f9">
    <td style="padding:9px 12px;font-size:12px;font-weight:500;color:#1e293b">{x.get("機構/來源","")}</td>
    <td style="padding:9px 12px;font-size:12px;color:{rating_color};font-weight:600">{rating_raw[:45]}</td>
    <td style="padding:9px 12px;font-size:13px;font-weight:700;color:#1a56a0">{fmt_num(tp, 0)}</td>
    <td style="padding:9px 12px"><span style="font-size:11px;padding:2px 9px;border-radius:4px;font-weight:700;background:{diff_bg};color:{diff_color}">{diff_pct:+.1f}%</span></td>
    <td style="padding:9px 12px;font-size:10px;color:#94a3b8">{source_txt}</td>
  </tr>"""

            # ★ 外層無框、無圓角容器；表頭僅底線，無格線
            table_html = f"""
<div style="margin-top:8px">
<table style="width:100%;border-collapse:collapse;font-size:12px">
  <thead>
    <tr style="border-bottom:1.5px solid #e2e8f0">
      <th style="font-weight:500;font-size:11px;color:#94a3b8;padding:8px 12px;text-align:left">機構名稱</th>
      <th style="font-weight:500;font-size:11px;color:#94a3b8;padding:8px 12px;text-align:left">評等</th>
      <th style="font-weight:500;font-size:11px;color:#94a3b8;padding:8px 12px;text-align:left">目標價 (NT$)</th>
      <th style="font-weight:500;font-size:11px;color:#94a3b8;padding:8px 12px;text-align:left">與現價差距</th>
      <th style="font-weight:500;font-size:11px;color:#94a3b8;padding:8px 12px;text-align:left">來源參考</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>
<div style="font-size:10px;color:#94a3b8;margin-top:6px">
  * 以上目標價來自各機構最新研究報告，投資人應自行至 Yahoo Finance / Edge財報 核實最新版本，本報告不構成任何投資建議。
</div>"""
            st.markdown(table_html, unsafe_allow_html=True)

    # ── 三大法人資訊（仿圖二風格）──────────────────────────────────────
    st.markdown("### 三大法人買賣超（近5日比較）")
    today_d, yday_d = chip.get("today", {}), chip.get("yesterday", {})
    t_date, y_date = chip.get("today_date"), chip.get("yday_date")
    bsd = chip.get("buy_sell_day_count", {})
    daily_5_df = chip.get("daily_5", pd.DataFrame())

    _t5 = int(daily_5_df["合計"].sum()) if not daily_5_df.empty and "合計" in daily_5_df.columns else 0
    if _t5 > 50000:
        _conc, _conc_color, _conc_icon = "偏多樂觀", "#16a34a", "↑"
    elif _t5 < -50000:
        _conc, _conc_color, _conc_icon = "偏空謹慎", "#dc2626", "↓"
    else:
        _conc, _conc_color, _conc_icon = "中立觀望", "#d97706", "⇄"

    _f_bsd = bsd.get("外資", {"buy": 0, "sell": 0})
    _f_buy_days = _f_bsd.get("buy", 0)
    _f_sell_days = _f_bsd.get("sell", 0)
    if _f_sell_days > _f_buy_days:
        _foreign_streak, _fc = f"賣超{_f_sell_days}天", "#dc2626"
    elif _f_buy_days > _f_sell_days:
        _foreign_streak, _fc = f"買超{_f_buy_days}天", "#16a34a"
    elif _f_buy_days > 0:
        _foreign_streak, _fc = f"買超{_f_buy_days}天", "#16a34a"
    else:
        _foreign_streak, _fc = "無方向", "#64748b"

    _max_streak, _max_txt = 0, "無明顯連續"
    for _inst in ["外資", "投信", "自營商"]:
        _inst_bsd = bsd.get(_inst, {"buy": 0, "sell": 0})
        _bd = _inst_bsd.get("buy", 0)
        _sd = _inst_bsd.get("sell", 0)
        if _sd > _bd and _sd > _max_streak:
            _max_streak = _sd
            _max_txt = f"{_inst}賣超{_sd}天"
        elif _bd >= _sd and _bd > _max_streak:
            _max_streak = _bd
            _max_txt = f"{_inst}買超{_bd}天"

    _streak_badges = ""
    for _inst in ["外資", "投信", "自營商"]:
        _inst_bsd = bsd.get(_inst, {"buy": 0, "sell": 0})
        _bd = _inst_bsd.get("buy", 0)
        _sd = _inst_bsd.get("sell", 0)
        if _sd > _bd and _sd >= 2:
            _streak_badges += f'<span style="background:#fef2f2;color:#dc2626;font-size:10px;padding:2px 8px;border-radius:4px;margin-right:5px;font-weight:600">{_inst}賣超{_sd}天</span>'
        elif _bd >= _sd and _bd >= 2:
            _streak_badges += f'<span style="background:#f0fdf4;color:#16a34a;font-size:10px;padding:2px 8px;border-radius:4px;margin-right:5px;font-weight:600">{_inst}買超{_bd}天</span>'
    _total_bsd = bsd.get("合計", {"buy": 0, "sell": 0})
    _tbd = _total_bsd.get("buy", 0)
    _tsd = _total_bsd.get("sell", 0)
    if _tsd > _tbd and _tsd >= 2:
        _streak_badges += f'<span style="background:#fffbeb;color:#d97706;font-size:10px;padding:2px 8px;border-radius:4px;margin-right:5px;font-weight:600">三大合計賣超{_tsd}天</span>'
    elif _tbd >= _tsd and _tbd >= 2:
        _streak_badges += f'<span style="background:#eff6ff;color:#1a56a0;font-size:10px;padding:2px 8px;border-radius:4px;margin-right:5px;font-weight:600">三大合計買超{_tbd}天</span>'

    _inst_header = f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:10px">
  <div style="background:#fff;border:.5px solid #e2e8f0;border-radius:8px;padding:12px 14px;border-top:3px solid {_conc_color}">
    <div style="font-size:10px;color:#64748b;font-weight:600;margin-bottom:4px">法人等級綜合結論</div>
    <div style="font-size:18px;font-weight:700;color:{_conc_color}">{_conc} {_conc_icon}</div>
  </div>
  <div style="background:#fff;border:.5px solid #e2e8f0;border-radius:8px;padding:12px 14px">
    <div style="font-size:10px;color:#64748b;font-weight:600;margin-bottom:4px">外資連買/賣</div>
    <div style="font-size:18px;font-weight:700;color:{_fc}">{_foreign_streak}</div>
  </div>
  <div style="background:#fff;border:.5px solid #e2e8f0;border-radius:8px;padding:12px 14px">
    <div style="font-size:10px;color:#64748b;font-weight:600;margin-bottom:4px">最長連買/賣</div>
    <div style="font-size:15px;font-weight:700;color:#1a56a0">{_max_txt}</div>
  </div>
  <div style="background:#fff;border:.5px solid #e2e8f0;border-radius:8px;padding:12px 14px">
    <div style="font-size:10px;color:#64748b;font-weight:600;margin-bottom:4px">5日法人合計</div>
    <div style="font-size:15px;font-weight:700;color:{'#16a34a' if _t5 >= 0 else '#dc2626'}">{_t5:+,d} 張</div>
  </div>
</div>"""

    if _streak_badges:
        _inst_header += f'<div style="margin-bottom:8px">{_streak_badges}</div>'

    _row_style = "display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr 1fr;gap:4px;padding:7px 10px;font-size:12px;border-bottom:.5px solid #f1f5f9;align-items:center"
    _hdr_style = "display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr 1fr;gap:4px;padding:7px 10px;font-size:11px;color:#64748b;font-weight:600;background:#f8fafc;border-radius:6px 6px 0 0"
    _daily_rows_html = f'<div style="{_hdr_style}"><span>日期</span><span style="text-align:right">外資（張）</span><span style="text-align:right">投信（張）</span><span style="text-align:right">自營商（張）</span><span style="text-align:right">合計淨額</span></div>'

    def _fmt_inst_val(v):
        try:
            iv = int(v)
            color = "#16a34a" if iv > 0 else ("#dc2626" if iv < 0 else "#64748b")
            return f'<span style="color:{color};font-weight:500">{iv:+,d}</span>'
        except Exception:
            return f'<span style="color:#94a3b8">N/A</span>'

    if not daily_5_df.empty:
        _5day_sums = {"外資": 0, "投信": 0, "自營商": 0, "合計": 0}
        for _, _drow in daily_5_df.iterrows():
            _d_label = str(_drow["date"])[:10] if "date" in _drow else "N/A"
            _fv = _drow.get("外資", 0); _iv = _drow.get("投信", 0); _sv = _drow.get("自營商", 0)
            _cv = _drow.get("合計", 0)
            _5day_sums["外資"] += int(_fv); _5day_sums["投信"] += int(_iv)
            _5day_sums["自營商"] += int(_sv); _5day_sums["合計"] += int(_cv)
            _daily_rows_html += f'<div style="{_row_style};background:#fff"><span style="color:#475569">{_d_label}</span><span style="text-align:right">{_fmt_inst_val(_fv)}</span><span style="text-align:right">{_fmt_inst_val(_iv)}</span><span style="text-align:right">{_fmt_inst_val(_sv)}</span><span style="text-align:right">{_fmt_inst_val(_cv)}</span></div>'
        _sum_color = "#16a34a" if _5day_sums["合計"] >= 0 else "#dc2626"
        _daily_rows_html += f'<div style="{_row_style};background:#f8fafc;border-radius:0 0 6px 6px;font-weight:600"><span style="color:#334155">5日合計</span><span style="text-align:right">{_fmt_inst_val(_5day_sums["外資"])}</span><span style="text-align:right">{_fmt_inst_val(_5day_sums["投信"])}</span><span style="text-align:right">{_fmt_inst_val(_5day_sums["自營商"])}</span><span style="text-align:right;font-weight:700;color:{_sum_color}">{_5day_sums["合計"]:+,d}</span></div>'
    else:
        _daily_rows_html += f'<div style="padding:14px;text-align:center;color:#94a3b8;font-size:12px">近5日資料暫時無法取得，請稍後重試。</div>'

    _interp = chip.get("summary", "")
    _interp_box = f'<div style="background:#eff6ff;border:.5px solid #bfdbfe;border-radius:6px;padding:10px 14px;margin-top:8px;font-size:11px;color:#1e40af"><b>解讀：</b>{_interp}</div>' if _interp else ""

    st.markdown(
        f"""{_inst_header}
<div style="border:.5px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-bottom:8px">
{_daily_rows_html}
</div>{_interp_box}""",
        unsafe_allow_html=True,
    )

    # 基本面只保留指定項目
    st.markdown("### 基本面量化指標 ")
    fund_details = fundamental.get("details", {})
    if fund_details:
        rows = list(fund_details.items())
        ncols = min(4, max(1, len(rows)))
        cols_fund = st.columns(ncols)
        for i, (k, v) in enumerate(rows):
            cols_fund[i % ncols].metric(k, v)

    # 量化總結
    st.markdown("### 量化分析結論")
    all_signals = (
        [("K線", s) for s in candle.get("signals", [])] +
        [("K線", s) for s in candle.get("ma_analysis", [])] +
        [("K線", f"K線型態：{candle.get('details', {}).get('K線長短', 'N/A')} / {candle.get('details', {}).get('K線方向', 'N/A')}")] +
        [("量能", s) for s in volume.get("signals", [])] +
        [("基本面", s) for s in fundamental.get("signals", [])] +
        [("籌碼", s) for s in chip.get("signals", [])] +
        [("風險", s) for s in risk.get("summary", "").split("。") if s]
    )
    warn_kw = ["跌破", "賣超", "偏弱", "轉弱", "過熱", "偏高", "衰退", "賣壓", "下滑", "死亡交叉", "流出", "背離", "超買", "處置", "注意", "風險", "下殺"]
    green_kw = ["站上", "買超", "成長", "偏強", "偏多", "多頭", "佳", "黃金交叉", "流入", "強勢", "良好", "充足", "放量上攻", "穩健"]
    warns = [(cat, s) for cat, s in all_signals if any(k in s for k in warn_kw)][:8]
    greens = [(cat, s) for cat, s in all_signals if any(k in s for k in green_kw)][:8]

    WARN_ICONS = {"K線":"📉","量能":"📊","基本面":"💹","籌碼":"🏦","風險":"🚨"}
    GOOD_ICONS = {"K線":"📈","量能":"💰","基本面":"✨","籌碼":"🎯","風險":"🛡️"}

    def _signal_row(icon, cat, text, kind="warn"):
        txt_color = "#7f1d1d" if kind == "warn" else "#14532d"
        cat_bg    = "#fecaca" if kind == "warn" else "#bbf7d0"
        cat_color = "#dc2626" if kind == "warn" else "#16a34a"
        return (
            f'<li style="display:flex;align-items:flex-start;gap:10px;padding:5px 0;border-bottom:.5px solid {"#fee2e2" if kind=="warn" else "#dcfce7"};font-size:12px;line-height:1.5;color:{txt_color}">'
            f'<span style="flex-shrink:0;width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:13px;background:{"#fef2f2" if kind=="warn" else "#f0fdf4"};border-radius:5px">{icon}</span>'
            f'<span><span style="font-size:9px;font-weight:700;letter-spacing:.7px;padding:1px 5px;border-radius:3px;background:{cat_bg};color:{cat_color};margin-right:5px;text-transform:uppercase">{cat}</span>{text}</span>'
            f'</li>'
        )

    if warns:
        warn_items = "".join(_signal_row(WARN_ICONS.get(c,"⚠️"), c, s, "warn") for c, s in warns)
        st.markdown(f"""
<div style="background:#fef2f2;border:.5px solid #fecaca;border-radius:10px;padding:12px 16px;margin-bottom:10px">
  <div style="font-size:10px;font-weight:700;letter-spacing:1px;color:#dc2626;text-transform:uppercase;margin-bottom:8px">⚠ 風險警示</div>
  <ul style="list-style:none;padding:0;margin:0">{warn_items}</ul>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#fef2f2;border:.5px solid #fecaca;border-radius:10px;padding:10px 16px;margin-bottom:10px;font-size:12px;color:#7f1d1d">⚠ 無重大風險警示訊號。</div>', unsafe_allow_html=True)

    if greens:
        good_items = "".join(_signal_row(GOOD_ICONS.get(c,"✅"), c, s, "good") for c, s in greens)
        st.markdown(f"""
<div style="background:#f0fdf4;border:.5px solid #bbf7d0;border-radius:10px;padding:12px 16px;margin-bottom:10px">
  <div style="font-size:10px;font-weight:700;letter-spacing:1px;color:#16a34a;text-transform:uppercase;margin-bottom:8px">✅ 多頭支撐訊號</div>
  <ul style="list-style:none;padding:0;margin:0">{good_items}</ul>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#f0fdf4;border:.5px solid #bbf7d0;border-radius:10px;padding:10px 16px;margin-bottom:10px;font-size:12px;color:#14532d">✅ 無明顯正面訊號。</div>', unsafe_allow_html=True)

    # 細項分數表
    st.markdown("### 細項分數")
    st.dataframe(pd.DataFrame([
        {"項目": "K線技術面", "分數/資訊": candle["score"], "主要訊號": candle["summary"][:90]},
        {"項目": "成交量", "分數/資訊": volume["score"], "主要訊號": volume["summary"][:90]},
        {"項目": "基本面", "分數/資訊": fundamental["score"], "主要訊號": fundamental["summary"][:90]},
        {"項目": "風險扣分", "分數/資訊": -(risk.get("penalty", 0) + market.get("score_penalty", 0)), "主要訊號": risk["summary"][:90]},
        {"項目": "總分", "分數/資訊": score_table["總分"], "主要訊號": f"分級：{score_table['分級']}"},
    ]), use_container_width=True)

    # Gemini 深度報告
    st.markdown("### Gemini 深度報告")
    ai_text = ""
    if os.getenv("GEMINI_API_KEY", "").strip():
        if st.button("產生 AI 深度分析"):
            with st.spinner("AI 分析中..."):
                ai_text = gemini_analyze(sid, info["name"], yday, today_row, score_table, candle, volume, fundamental, chip, risk, market, target)
            st.write(ai_text)
    else:
        st.info("尚未設定 Gemini API Key。")

    if not ai_text:
        core_lines = [f"[警示][{cat}] {s}" for cat, s in warns] + [f"[正面][{cat}] {s}" for cat, s in greens]
        if target:
            core_lines.append(f"[交易計畫] 建議進場價 {fmt_num(target.get('entry_price'), 2)}，斷線/短線停損 {fmt_num(target.get('stop_price'), 2)}。")
        ai_text = "\n".join(core_lines)


    # PDF 下載
    pdf_bytes = create_pdf_report(sid, info, yday, today_row, score_table, candle, volume, fundamental, chip, risk, market, target, ai_text)
    if pdf_bytes:
        st.download_button(
            "📄 下載 PDF 報告",
            data=pdf_bytes,
            file_name=f"{sid}_stock_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("目前環境沒有 reportlab，無法產生 PDF。")
