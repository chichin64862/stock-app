import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import requests
import urllib3
import io
import os
import time
from datetime import datetime
import urllib.request
import html

# 關閉 SSL 驗證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# PDF 函式庫
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- 1. 介面設定 ---
st.set_page_config(page_title="台股量化與價值分析終端", page_icon="Terminal", layout="wide", initial_sidebar_state="expanded")

# --- 2. CSS 風格 ---
st.markdown("""
<style>
    :root { color-scheme: dark !important; }
    .stApp { background-color: #06090F !important; }
    [data-testid="stSidebar"] { background-color: #0D131F !important; border-right: 1px solid #1E293B !important; }
    h1, h2, h3, p, span, div, label { font-family: 'Segoe UI', Tahoma, sans-serif; color: #E2E8F0 !important; }
    .stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] > div, input { background-color: #0D131F !important; color: white !important; border: 1px solid #1E293B !important; }
    .stMultiSelect span[data-baseweb="tag"] { background-color: #1E293B !important; color: #F8FAFC !important; }
    .stock-card { background-color: #0D131F !important; padding: 16px 24px; border-radius: 4px; border: 1px solid #1E293B !important; border-left: 4px solid #3B82F6 !important; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .card-header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid #1E293B !important; padding-bottom: 12px; margin-bottom: 16px; }
    .header-title { font-size: 1.4rem; font-weight: 700; color: #FFFFFF !important; letter-spacing: 1px; }
    .header-price { font-size: 1.4rem; font-weight: 700; color: #10B981 !important; margin-left: 16px; font-family: 'Consolas', monospace; }
    .quote-board { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background-color: #1E293B !important; border-radius: 4px; }
    .quote-item { background-color: #06090F !important; padding: 12px 16px; display: flex; flex-direction: column; justify-content: center; }
    .q-label { color: #64748B !important; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
    .q-val { color: #F8FAFC !important; font-weight: 700; font-size: 1.15rem; font-family: 'Consolas', 'Courier New', monospace; }
    .q-up { color: #10B981 !important; } .q-down { color: #EF4444 !important; } .q-neu { color: #F59E0B !important; }  
    .tag { padding: 3px 8px; border-radius: 2px; font-size: 0.75rem; font-weight: 700; margin-left: 8px; text-transform: uppercase; border: 1px solid; }
    .tag-moat { background-color: rgba(16, 185, 129, 0.1) !important; color: #10B981 !important; border-color: #10B981 !important; }
    .tag-risk { background-color: rgba(239, 68, 68, 0.1) !important; color: #EF4444 !important; border-color: #EF4444 !important; }
    .tag-logic { background-color: rgba(59, 130, 246, 0.1) !important; color: #3B82F6 !important; border-color: #3B82F6 !important; }
    .dcf-panel { background-color: #0D131F !important; border: 1px solid #334155 !important; border-left: 3px solid #8B5CF6 !important; padding: 12px 16px; margin-top: 16px; border-radius: 2px; }
    .ai-box { background-color: #0D131F !important; border: 1px solid #1E293B !important; border-left: 3px solid #3B82F6 !important; padding: 16px; margin-top: 16px; border-radius: 2px; font-size: 0.95rem; line-height: 1.6; color: #CBD5E1 !important; }
    .explain-box { background-color: #0F172A !important; border: 1px solid #334155 !important; border-left: 4px solid #F59E0B !important; padding: 16px 20px; border-radius: 4px; margin-bottom: 24px; color: #E2E8F0 !important; line-height: 1.7; font-size: 0.95rem; }
    .cmd-box { background-color: #1E293B !important; border: 1px solid #334155 !important; border-top: 4px solid #F59E0B !important; padding: 20px; border-radius: 4px; margin-bottom: 24px; }
    .stButton button, .stDownloadButton button { background-color: #1E293B !important; border: 1px solid #334155 !important; color: #F8FAFC !important; border-radius: 2px !important; font-weight: 600 !important; }
    .stButton button:hover, .stDownloadButton button:hover { border-color: #3B82F6 !important; color: #3B82F6 !important; background-color: #0D131F !important; }
    div.row-widget.stRadio > div { flex-direction: row; flex-wrap: wrap; gap: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 3. 初始化 ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'history_storage' not in st.session_state: st.session_state['history_storage'] = {}
if 'ai_results' not in st.session_state: st.session_state['ai_results'] = {}
if 'current_logic' not in st.session_state: st.session_state['current_logic'] = "Quant" 
if 'panel_page' not in st.session_state: st.session_state['panel_page'] = 1 
if 'list_page' not in st.session_state: st.session_state['list_page'] = 1 
if 'my_portfolio' not in st.session_state: st.session_state['my_portfolio'] = []

try: api_key = st.secrets["GEMINI_API_KEY"]
except Exception: api_key = None

# 【✅ 唯一修改處：加入檔案容量檢查，徹底解決 PDF 空白/亂碼問題】
@st.cache_resource
def setup_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    # 檢查字體檔案是否存在，且檔案大小是否正常 (避免下載失敗產生 0 byte 空檔案導致 PDF 空白)
    if not os.path.exists(font_path) or os.path.getsize(font_path) < 100000:
        try:
            r = requests.get("https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC-Regular.ttf", timeout=30)
            if r.status_code == 200:
                with open(font_path, 'wb') as f:
                    f.write(r.content)
        except: pass
    
    try: 
        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
        return 'ChineseFont'
    except: 
        try:
            pdfmetrics.registerFont(UnicodeCIDFont('MSung-Light'))
            return 'MSung-Light'
        except:
            return 'Helvetica'

font_name_global = setup_chinese_font()

# =========================================================================
# 💡 工具函數與核心數學引擎
# =========================================================================

def safe_float(val):
    try:
        if pd.isna(val) or val is None or str(val).strip() == '': return np.nan
        return float(str(val).replace(',', '').strip())
    except:
        return np.nan

def calc_beta(stock_ret, market_ret):
    aligned = stock_ret.align(market_ret, join='inner')
    if len(aligned[0]) < 10: return np.nan
    var = np.var(aligned[1])
    if var == 0: return np.nan
    cov = np.cov(aligned[0], aligned[1])[0][1]
    return cov / var

def calc_mdd(price_series):
    if price_series.empty: return np.nan
    cum_max = price_series.cummax()
    drawdown = (price_series - cum_max) / cum_max
    return drawdown.min()

def calc_sharpe(returns, rf=0.015):
    if returns.empty: return np.nan
    excess = returns - (rf / 252)
    return np.sqrt(252) * excess.mean() / excess.std()

def calculate_implied_growth(price, eps, r=0.10, terminal_g=0.02, years=10):
    if pd.isna(price) or pd.isna(eps) or eps <= 0 or price <= 0: return np.nan
    low, high = -0.99, 3.0 
    for _ in range(50): 
        mid = (low + high) / 2
        pv = sum([eps * ((1 + mid) ** t) / ((1 + r) ** t) for t in range(1, years + 1)])
        tv = (eps * ((1 + mid) ** years) * (1 + terminal_g)) / (r - terminal_g)
        calc_price = pv + tv / ((1 + r) ** years)
        diff = calc_price - price
        if abs(diff) < 0.01: return mid
        if diff > 0: high = mid
        else: low = mid
    return (low + high) / 2

def sanitize_data(df):
    if df.empty: return df
    if 'yield' in df.columns: 
        df['yield'] = df['yield'].apply(lambda x: x/100 if pd.notna(x) and x > 20 else x)
    return df

def calc_eps_yoy(fin, bal):
    try:
        if "Net Income" in fin.index and "Ordinary Shares Number" in bal.index:
            eps = fin.loc["Net Income"] / bal.loc["Ordinary Shares Number"]
            eps = eps.dropna()
            if len(eps) >= 2:
                return ((eps.iloc[0] - eps.iloc[1]) / abs(eps.iloc[1])) * 100
    except: pass
    return np.nan

def calc_peg(pe, eps_growth_pct):
    try:
        if pd.isna(pe) or pd.isna(eps_growth_pct) or eps_growth_pct == 0: return np.nan
        return pe / eps_growth_pct
    except: return np.nan

def get_revenue_yoy_mops(code):
    try:
        time.sleep(np.random.uniform(0.05, 0.15))
        url = "https://mops.twse.com.tw/mops/web/ajax_t05st10_ifrs"
        headers = {'User-Agent': 'Mozilla/5.0'}
        for typek in ["sii", "otc"]:
            payload = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "TYPEK": typek, "co_id": code}
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text: continue
            tables = pd.read_html(io.StringIO(res.text))
            if len(tables) > 0:
                df = tables[0]
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
                df = df.sort_values(df.columns[0], ascending=False)
                if len(df) >= 2 and "營業收入" in df.columns:
                    rev_now = safe_float(df.iloc[0]["營業收入"])
                    rev_prev = safe_float(df.iloc[1]["營業收入"])
                    if rev_prev != 0 and pd.notna(rev_now) and pd.notna(rev_prev): 
                        return (rev_now - rev_prev) / abs(rev_prev)
    except: pass
    return np.nan

# =========================================================================
# 💡 資料前置預載引擎
# =========================================================================

@st.cache_data(ttl=86400)
def get_tw_stock_list():
    stock_map, industry_map, code_to_industry = {}, {}, {}
    custom_sector_override = {
        "2330": "半導體業", "2454": "半導體業", "2303": "半導體業", "3711": "半導體業",
        "2382": "電腦及週邊設備業", "3231": "電腦及週邊設備業", "2356": "電腦及週邊設備業", "2376": "電腦及週邊設備業", "6669": "電腦及週邊設備業",
        "2308": "電子零組件業", "3008": "光電業", "2327": "電子零組件業", "2383": "電子零組件業",
        "2317": "其他電子業", "2881": "金融保險業", "2603": "航運業"
    }
    
    twse_ind_map = {
        "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維", "05": "電機機械",
        "06": "電器電纜", "07": "化學工業", "08": "玻璃陶瓷", "09": "造紙工業", "10": "鋼鐵工業",
        "11": "橡膠工業", "12": "汽車工業", "14": "建材營造", "15": "航運業", "16": "觀光餐旅",
        "17": "金融保險", "18": "貿易百貨", "19": "綜合", "20": "其他產業", "21": "化學工業",
        "22": "生技醫療業", "23": "油電燃氣業", "24": "半導體業", "25": "電腦及週邊設備業",
        "26": "光電業", "27": "通信網路業", "28": "電子零組件業", "29": "電子通路業",
        "30": "資訊服務業", "31": "其他電子業", "32": "文化創意業", "33": "農業科技業",
        "34": "電子商務業", "80": "管理股票",
        "1": "水泥工業", "2": "食品工業", "3": "塑膠工業", "4": "紡織纖維", "5": "電機機械",
        "6": "電器電纜", "7": "化學工業", "8": "玻璃陶瓷", "9": "造紙工業"
    }

    try:
        import twstock
        for code, info in twstock.codes.items():
            if info.type in ['股票', 'ETF']:  
                full = f"{code}.TW" if info.market == '上市' else f"{code}.TWO"
                name = str(info.name).strip()
                ind_raw = str(info.group).strip() if info.group else ""
                
                ind = twse_ind_map.get(ind_raw, ind_raw)
                if not ind or ind.isdigit() or ind.lower() == "none": ind = "其他產業"
                
                group = custom_sector_override.get(code, ind)
                if code.startswith('00'): group = 'ETF'
                
                stock_map[full] = f"{full} {name}"
                if group not in industry_map: industry_map[group] = []
                industry_map[group].append(full)
                code_to_industry[code] = group
    except: pass

    for mkt_url, sfx in [("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW"), 
                         ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO")]:
        try:
            res = requests.get(mkt_url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=5)
            if res.status_code == 200:
                for item in res.json():
                    code = str(item.get('公司代號', '')).strip()
                    name = str(item.get('公司簡稱', '')).strip()
                    ind_raw = str(item.get('產業別', '其他')).strip()
                    
                    ind = twse_ind_map.get(ind_raw, ind_raw)
                    if not ind or ind.isdigit(): ind = "其他產業"
                    
                    if len(code) == 4 and code.isdigit():
                        full = f"{code}{sfx}"
                        group = custom_sector_override.get(code, ind)
                        if code.startswith('00'): group = 'ETF'
                        
                        if full not in stock_map:
                            stock_map[full] = f"{full} {name}"
                            if group not in industry_map: industry_map[group] = []
                            industry_map[group].append(full)
                            code_to_industry[code] = group
        except: pass

    return stock_map, industry_map, code_to_industry

stock_map, industry_map, code_to_industry_map = get_tw_stock_list()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_global_market_data():
    twse_data = {}
    rev_data = {}
    mkt_ret = pd.Series(dtype=float)
    
    for mkt_url in ["https://openapi.twse.com.tw/v1/opendata/t187ap14_L", "https://www.tpex.org.tw/openapi/v1/t187ap14_O"]:
        try:
            res = requests.get(mkt_url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=10)
            if res.status_code == 200:
                for item in res.json():
                    pe = safe_float(item.get("本益比"))
                    pb = safe_float(item.get("股價淨值比"))
                    yld = safe_float(item.get("殖利率(%)"))
                    twse_data[str(item.get("證券代號"))] = {
                        "pe": pe if pd.notna(pe) and pe > 0 else np.nan,
                        "pb": pb if pd.notna(pb) else np.nan,
                        "yield": yld / 100.0 if pd.notna(yld) else np.nan
                    }
        except: pass

    for rev_url in ["https://openapi.twse.com.tw/v1/opendata/t187ap05_L", "https://www.tpex.org.tw/openapi/v1/t187ap05_O"]:
        try:
            res_rev = requests.get(rev_url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=10)
            if res_rev.status_code == 200:
                for item in res_rev.json():
                    yoy_str = item.get("去年同月增減(%)", item.get("營業收入-去年同月增減(%)"))
                    yoy = safe_float(yoy_str)
                    code_str = str(item.get("公司代號", "")).strip()
                    if pd.notna(yoy) and code_str:
                        rev_data[code_str] = yoy / 100.0
        except: pass

    try:
        market = yf.Ticker("^TWII").history(period="6mo")
        if not market.empty: mkt_ret = market['Close'].pct_change().dropna()
    except: pass

    return twse_data, rev_data, mkt_ret

def get_safe_val(df, keys, col_idx=0):
    for k in keys:
        if k in df.index:
            val = df.loc[k].iloc[col_idx]
            return float(val) if pd.notna(val) else np.nan
    return np.nan

# =========================================================================
# 💡 核心自研數據庫 (多源聚合)
# =========================================================================

def get_stock_full_optimized(stock_str, global_twse, global_rev, mkt_ret):
    code = stock_str.split(' ')[0].split('.')[0]
    name = code
    if stock_str in stock_map:
        parts = stock_map[stock_str].split(' ')
        if len(parts) > 1: name = parts[1]
    elif len(stock_str.split(' ')) > 1:
        name = stock_str.split(' ')[1]
        
    symbol = stock_str.split(' ')[0]
    if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): symbol += '.TW'
    
    ticker = yf.Ticker(symbol)
    
    data = {
        '代號': code, '名稱': name, 'close_price': np.nan, 'pe': np.nan, 'pb': np.nan, 'yield': np.nan,
        'roe': np.nan, 'gross_margins': np.nan, 'eps_growth': np.nan, 'rev_growth': np.nan, 
        'beta': np.nan, 'fcf': np.nan, 'fcf_yield': np.nan, 'de_ratio': np.nan, 'market_cap': np.nan,
        'sharpe': np.nan, 'mdd': np.nan, 'volatility': np.nan, 'peg': np.nan, 'priceToMA60': np.nan, 
        'implied_growth': np.nan, 'chips': 0, 'industry': code_to_industry_map.get(code, '未分類'), 
        'full_symbol': stock_str, 'history': pd.DataFrame()
    }
    if code.startswith('00'): data['industry'] = 'ETF'
    
    try:
        hist = ticker.history(period="6mo")
        if not hist.empty:
            price = hist['Close']
            returns = price.pct_change().dropna()
            current_price = float(price.iloc[-1])
            data['close_price'] = current_price
            data['volatility'] = returns.std() * np.sqrt(252)
            data['mdd'] = calc_mdd(price)
            data['sharpe'] = calc_sharpe(returns)
            data['history'] = hist
            
            ma60 = price.rolling(60).mean().iloc[-1]
            if pd.notna(ma60) and ma60 > 0: data["priceToMA60"] = (current_price / ma60) - 1
            if not mkt_ret.empty: data["beta"] = calc_beta(returns, mkt_ret)
    except: pass

    try:
        info = ticker.info
        data["pe"] = info.get("trailingPE", np.nan)
        data["roe"] = info.get("returnOnEquity", np.nan)
        data["gross_margins"] = info.get("grossMargins", np.nan)
        data["fcf"] = info.get("freeCashflow", np.nan)
        data["market_cap"] = info.get("marketCap", np.nan)
        if pd.isna(data["beta"]): data["beta"] = info.get("beta", np.nan)
    except: pass

    try:
        fin = ticker.quarterly_financials
        bal = ticker.quarterly_balance_sheet
        cf = ticker.quarterly_cashflow

        if np.isnan(data["roe"]):
            try: data["roe"] = float(fin.loc["Net Income"].iloc[0] * 4 / bal.loc["Total Stockholder Equity"].iloc[0])
            except: pass

        if np.isnan(data["gross_margins"]):
            try: data["gross_margins"] = float(fin.loc["Gross Profit"].iloc[0] / fin.loc["Total Revenue"].iloc[0])
            except: pass

        if np.isnan(data["fcf"]):
            try:
                op = cf.loc["Total Cash From Operating Activities"].iloc[0] if "Total Cash From Operating Activities" in cf.index else cf.loc["Operating Cash Flow"].iloc[0]
                capex = cf.loc["Capital Expenditures"].iloc[0] if "Capital Expenditures" in cf.index else cf.loc["Capital Expenditure"].iloc[0]
                data["fcf"] = float(op + capex)
            except: pass

        data["eps_growth"] = calc_eps_yoy(fin, bal)
        
        if pd.isna(data["market_cap"]):
            try:
                shares = bal.loc["Ordinary Shares Number"].iloc[0]
                if pd.notna(data["close_price"]) and data["close_price"] > 0:
                    data["market_cap"] = data["close_price"] * shares
            except: pass
            
        try: data["de_ratio"] = float(bal.loc["Total Debt"].iloc[0] / bal.loc["Total Stockholder Equity"].iloc[0])
        except: pass
    except: pass

    try:
        if not np.isnan(data["fcf"]) and not np.isnan(data["market_cap"]) and data["market_cap"] > 0:
            data["fcf_yield"] = (data["fcf"] * 4) / data["market_cap"]
    except: pass

    if pd.isna(data['roe']) or pd.isna(data['gross_margins']):
        try:
            wg_url = f"https://www.wantgoo.com/investrue/api/v1/stock/{code}/financial-ratios"
            wg_headers = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
            res_wg = requests.get(wg_url, headers=wg_headers, timeout=5)
            if res_wg.status_code == 200:
                wg_json = res_wg.json()
                if isinstance(wg_json, list) and len(wg_json) > 0:
                    if pd.isna(data['roe']): data['roe'] = safe_float(wg_json[0].get('returnOnEquity')) / 100.0
                    if pd.isna(data['gross_margins']): data['gross_margins'] = safe_float(wg_json[0].get('grossMargin')) / 100.0
        except: pass

    if code in global_twse:
        tw_pe = global_twse[code].get("pe", np.nan)
        if pd.notna(tw_pe): data["pe"] = tw_pe 
        
        tw_pb = global_twse[code].get("pb", np.nan)
        if pd.notna(tw_pb): data["pb"] = tw_pb
        
        tw_yld = global_twse[code].get("yield", np.nan)
        if pd.notna(tw_yld): data["yield"] = tw_yld

    if pd.isna(data["pe"]) or data["pe"] <= 0:
        try:
            if "Net Income" in fin.index and "Ordinary Shares Number" in bal.index:
                ni_ttm = fin.loc["Net Income"].iloc[:4].sum()
                shares = bal.loc["Ordinary Shares Number"].iloc[0]
                if shares > 0 and pd.notna(data["close_price"]):
                    eps_ttm = ni_ttm / shares
                    if eps_ttm != 0:
                        data["pe"] = data["close_price"] / eps_ttm
        except: pass

    if pd.isna(data["yield"]):
        try:
            divs = ticker.dividends
            if not divs.empty:
                last_year_div = divs[divs.index > (pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365))].sum()
                if pd.notna(data['close_price']) and data['close_price'] > 0:
                    data["yield"] = last_year_div / data['close_price']
        except: pass

    if code in global_rev:
        data["rev_growth"] = global_rev[code]
    elif pd.isna(data["rev_growth"]) or data["rev_growth"] == 0:
        mo_rev = get_revenue_yoy_mops(code)
        if pd.notna(mo_rev): data["rev_growth"] = mo_rev

    data["peg"] = calc_peg(data["pe"], data["eps_growth"])
    
    if pd.notna(data["close_price"]) and pd.notna(data["pe"]) and data["pe"] > 0:
        data["implied_growth"] = calculate_implied_growth(data["close_price"], data["close_price"] / data["pe"])
        
    for key in ['roe', 'gross_margins', 'rev_growth', 'fcf_yield', 'yield', 'de_ratio']:
        if pd.notna(data[key]): data[key] *= 100

    return data

def batch_scan_stocks(stock_list):
    results = []
    history_map = {}
    twse_data, rev_data, mkt_ret = fetch_global_market_data()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_full_optimized, s, twse_data, rev_data, mkt_ret): s for s in stock_list}
        for future in concurrent.futures.as_completed(future_to_stock):
            try:
                data = future.result()
                if data and pd.notna(data['close_price']):
                    history_map[data['代號']] = data.pop('history')
                    results.append(data)
            except: continue
            
    df = pd.DataFrame(results)
    return df, history_map

# --- 8. 評分邏輯 ---
def calculate_score(df, logic_type="Quant"):
    if df.empty: return df, None
    required_cols = ['pe', 'pb', 'yield', 'rev_growth', 'eps_growth', 'gross_margins', 'fcf_yield', 'de_ratio', 'beta', 'sharpe', 'mdd', 'implied_growth', 'peg', 'volatility', 'roe', 'priceToMA60']
    for col in required_cols:
        if col not in df.columns: df[col] = np.nan
    
    df_norm = df.copy()
    scores, plans, quality_tags = [], [], []
    calc_df = df.fillna({'pe': 50, 'volatility': 0.5, 'beta': 1.0, 'de_ratio': 100})

    for idx, row in calc_df.iterrows():
        total_score = total_weight = 0
        if logic_type == "Quant":
            config = {'Sharpe': {'col': 'sharpe', 'dir': 'max', 'w': 4.0, 'cat': '動能'}, 'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 2.0, 'cat': '風險'}, 'Beta': {'col': 'beta', 'dir': 'mid', 'w': 1.0, 'cat': '風險'}}
        else:
            config = {'ROE': {'col': 'roe', 'dir': 'max', 'w': 2.0, 'cat': '財報'}, 'GrossMargin': {'col': 'gross_margins', 'dir': 'max', 'w': 1.5, 'cat': '財報'}, 'FCF_Yield': {'col': 'fcf_yield', 'dir': 'max', 'w': 2.0, 'cat': '價值'}, 'DE_Ratio': {'col': 'de_ratio', 'dir': 'min', 'w': 1.5, 'cat': '風險'}, 'EPS_Growth': {'col': 'eps_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'}}
            
        for name, setting in config.items():
            val = calc_df.loc[idx, setting['col']]
            all_vals = calc_df[setting['col']].dropna()
            if all_vals.empty: rank = 0.5
            else: rank = (all_vals < val).mean() if setting['dir'] == 'max' else (all_vals > val).mean()
            norm = rank if setting['dir'] in ['max', 'min'] else 1 - abs(rank - 0.5)*2
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm * 100
            total_score += norm * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final, 1))
        
        sh, vol, b, mdd = row.get('sharpe', 0), row.get('volatility', 1), row.get('beta', 1.0), row.get('mdd', 0)
        if pd.isna(b): b = 1.0
        
        q_tag = ""
        if logic_type == "Quant":
            if mdd < -0.25: q_tag = "回撤過大 (剔除)"; plans.append("跌破 25% 防線"); scores[-1] = 0 
            elif sh > 1.0 and b >= 1.0: q_tag = "成長型資產"; plans.append("配置主動能")
            elif sh > 1.0 and b < 1.0: q_tag = "防禦型資產"; plans.append("配置穩定基石")
            elif sh > 0 and vol < 0.25: q_tag = "中性資產"; plans.append("降低組合波動")
            elif sh < 0: q_tag = "風險報酬不對等"; plans.append("避開標的")
            else: q_tag = "中立觀望"; plans.append("中立觀望")
        else: 
            roe, gm, de, fcf = row.get('roe', 0), row.get('gross_margins', 0), row.get('de_ratio', 100), row.get('fcf_yield', 0)
            if mdd < -0.25: q_tag = "回撤過大 (剔除)"; plans.append("跌破 25% 防線"); scores[-1] = 0
            elif roe > 15 and gm > 40 and de < 100 and fcf > 0: q_tag = "護城河優良"; plans.append("價值浮現")
            elif de > 200 or fcf < -5: q_tag = "財務風險"; plans.append("避開標的")
            else: q_tag = "中立觀望"; plans.append("合理估值" if final > 60 else "中立觀望")
            
        quality_tags.append(q_tag)
            
    df['Score'] = scores
    df['Strategy'] = plans
    df['Quality'] = quality_tags
    return df.sort_values('Score', ascending=False), df_norm

def get_radar_data(df_norm_row):
    cats = {'價值估值': 0, '成長動能': 0, '趨勢強度': 0, '風險控管': 0, '財務體質': 0}
    counts = {'價值估值': 0, '成長動能': 0, '趨勢強度': 0, '風險控管': 0, '財務體質': 0}
    map_dict = {'價值': '價值估值', '成長': '成長動能', '動能': '趨勢強度', '風險': '風險控管', '財報': '財務體質'}
    for col in df_norm_row.index:
        if str(col).endswith('_n'):
            cat_raw = str(col).split('_')[0]
            cat = map_dict.get(cat_raw, cat_raw)
            if cat in cats:
                cats[cat] += df_norm_row[col]
                counts[cat] += 1
    radar = {}
    for k, v in cats.items(): radar[k] = v / counts[k] if counts[k] > 0 else 50
    return radar

def plot_radar_chart_ui(title, radar_data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(radar_data.values()), theta=list(radar_data.keys()), fill='toself', name=title, line_color='#3B82F6', fillcolor='rgba(59, 130, 246, 0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=30, r=30), height=250, font=dict(color='#9CA3AF'))
    return fig

def plot_trend_dashboard(title, history_df, ma_bias):
    if history_df is None or history_df.empty: return None
    history_df['MA60'] = history_df['Close'].rolling(window=60).mean()
    current_price = history_df['Close'].iloc[-1]
    bias_pct = ma_bias * 100
    if bias_pct > 15: status_text = f"過熱區間"
    elif bias_pct > 5: status_text = f"多頭強勢"
    elif bias_pct > -5: status_text = f"整理區間"
    elif bias_pct > -15: status_text = f"超跌價值"
    else: status_text = f"空頭弱勢"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='Price', line=dict(color='#3B82F6', width=2)))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['MA60'], name='MA60', line=dict(color='#F59E0B', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=[history_df.index[-1]], y=[current_price], mode='markers', marker=dict(color='#10B981', size=8), showlegend=False))
    fig.update_layout(title=dict(text=f"<b>中長期趨勢研判</b><br><span style='font-size:12px; color:#9CA3AF'>季線乖離 {bias_pct:.1f}% | {status_text}</span>", font=dict(color='#F8FAFC', size=14), y=0.95), xaxis=dict(showgrid=False, linecolor='#334155', tickfont=dict(color='#64748B')), yaxis=dict(showgrid=True, gridcolor='#1E293B', tickfont=dict(color='#64748B')), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=60, b=20, l=0, r=0), height=250, showlegend=False, hovermode="x unified")
    return fig

def get_tag_explanation(row, logic_type):
    tag = row.get('Quality', '')
    sharpe = row.get('sharpe', 0)
    roe = row.get('roe', 0)
    gm = row.get('gross_margins', 0)
    fcf = row.get('fcf_yield', 0)
    de = row.get('de_ratio', 0)
    mdd = row.get('mdd', 0)
    
    if tag == "回撤過大 (剔除)": return f"**為什麼被系統剔除？**<br>此標的過去半年內的最大回撤 (MDD) 高達 **{mdd*100:.1f}%**。依據嚴格的量化風控紀律，任何跌破 25% 容忍防線的資產，無論其基本面或夏普值多麼亮眼，都可能造成投資組合的永久性資本損傷，因此被系統強制剔除。"
    if logic_type == "Quant":
        if tag == "成長型資產": return f"**為什麼被系統評定為「🚀 成長型資產」？**<br>此標的近期年化夏普值高達 **{sharpe:.2f}** 且 Beta 大於 1。依據林哲群教授《紀律長贏》觀念，其報酬率累積速度遠大於風險的增加，屬於位於效率前緣的高回報攻擊動能。"
        elif tag == "防禦型資產": return f"**為什麼被系統評定為「🛡️ 防禦型 নিষ্ঠ」？**<br>此標的近期年化夏普值高達 **{sharpe:.2f}**，且 Beta 值小於 1。具備低於大盤的波動特性與高性價比報酬，是優化投資組合下檔風險的防禦基石。"
        elif tag == "中性資產": return f"**為什麼被系統評定為「⚖️ 中性資產」？**<br>此標的夏普值大於 0 且波動率低於 25%。屬於走勢平穩、與大盤關聯較弱的穩定器，能有效稀釋整體 MPT 投資組合的共變異數波動。"
        elif tag == "風險報酬不對等": return f"**為什麼被系統評定為「⚠️ 風險報酬不對等」？**<br>此標的夏普值為 **{sharpe:.2f}**（小於 0）。投資此類資產並未帶來與風險相匹配的回報，不符合長期紀律投資精神。"
        else: return "**標籤說明**<br>該標的各項量化指標落在市場均值區間，風險與報酬呈現中性，未觸發極端強勢或弱勢的演算法警示。"
    else:
        if tag == "護城河優良": return f"**為什麼被系統評定為「護城河優良」？**<br>此標的完美符合巴菲特的核心護城河條件：<br>1. **高資本效率**：ROE 達 **{roe:.1f}%**。<br>2. **強大定價權**：毛利率達 **{gm:.1f}%**。<br>3. **真實變現力**：自由現金流收益率為正且無過度舉債。"
        elif tag == "財務風險": return f"**為什麼被系統評定為「財務風險」？**<br>系統偵測到其負債權益比高達 **{de:.1f}%**，或現金流低至 **{fcf:.2f}%**。代表該公司可能正在靠過度舉債擴張，具有高度「虛胖」風險。"
        else: return "**標籤說明**<br>該標的在財務體質上表現中規中矩，雖未達嚴苛的「絕對護城河」標準，但也無明顯的財務致命傷。"

def greedy_mpt_optimization(pool_df, returns_df, target_n, metric_col, initial_selected=None, target_vol_max=0.15):
    selected_codes = list(initial_selected) if initial_selected else []
    candidates = pool_df['代號'].tolist()
    for c in selected_codes:
        if c in candidates: candidates.remove(c)
    if not candidates: return selected_codes
    max_metric = pool_df[metric_col].max()
    min_metric = pool_df[metric_col].min()
    if max_metric == min_metric: max_metric = min_metric + 1 
    
    if not selected_codes and len(candidates) > 0:
        first_code = pool_df.loc[pool_df['代號'].isin(candidates), metric_col].idxmax()
        first_code = pool_df.loc[first_code, '代號']
        selected_codes.append(first_code)
        candidates.remove(first_code)
        
    while len(selected_codes) < (len(initial_selected or []) + target_n) and candidates:
        best_score = -999999
        best_code = None
        for code in candidates:
            base_val = pool_df[pool_df['代號'] == code][metric_col].values[0]
            norm_metric = (base_val - min_metric) / (max_metric - min_metric)
            corrs = [returns_df[code].corr(returns_df[sel_code]) for sel_code in selected_codes if code in returns_df.columns and sel_code in returns_df.columns]
            corrs = [c for c in corrs if not pd.isna(c)]
            avg_corr = np.mean(corrs) if corrs else 0
            max_corr = max(corrs) if corrs else 0
            corr_penalty = (avg_corr * 5.0) + (max(0, max_corr - 0.4) * 10.0)
            
            test_codes = selected_codes + [code]
            valid_codes = [c for c in test_codes if c in returns_df.columns]
            port_vol = 0.20 
            if len(valid_codes) > 1:
                cov_matrix = returns_df[valid_codes].cov() * 252
                weights = []
                for tc in valid_codes:
                    t_tag = pool_df[pool_df['代號'] == tc]['戰略定位'].values[0] if len(pool_df[pool_df['代號'] == tc])>0 else ""
                    ts = pool_df[pool_df['代號'] == tc]['sharpe'].values[0] if len(pool_df[pool_df['代號'] == tc])>0 else 0
                    tv = pool_df[pool_df['代號'] == tc]['volatility'].values[0] if len(pool_df[pool_df['代號'] == tc])>0 else 1
                    if t_tag == "🚀 成長型資產": weights.append(0.135 if (ts > 2.0 and tv < 0.30) else 0.10)
                    elif t_tag == "🛡️ 防禦型資產": weights.append(0.10)
                    else: weights.append(0.065)
                w = np.array(weights)
                if np.sum(w) > 0: w = w / np.sum(w)
                port_variance = np.dot(w.T, np.dot(cov_matrix, w))
                port_vol = np.sqrt(port_variance)
                
            vol_penalty = (port_vol - target_vol_max) * 100.0 if port_vol > target_vol_max else 0
            obj_score = norm_metric - corr_penalty - vol_penalty 
            if obj_score > best_score:
                best_score = obj_score
                best_code = code
        if best_code:
            selected_codes.append(best_code)
            candidates.remove(best_code)
        else: break
    return selected_codes

def generate_custom_signal(row, regime):
    s, mdd, b, tag = row.get('sharpe', 0), row.get('mdd', 0), row.get('beta', 1.0), row.get('Quality', '')
    if mdd < -0.25: return "🛑 強制停損 (MDD破防)"
    if s < 0: return "⚠️ 劣勢資產 (建議剔除)"
    if regime == "Bull":
        if tag == "成長型資產" and s > 2.0 and mdd > -0.10: return "🔥 強勢加碼 (放大獲利)"
        if tag in ["防禦型資產", "中性資產"]: return "📉 逢高減碼 (釋放資金)"
        return "⚖️ 持有續抱"
    else: 
        if b > 1.2: return "📉 降持高波 (收回現金)"
        if tag in ["防禦型資產", "中性資產"]: return "🛡️ 逢低加碼 (鞏固底盤)"
        return "⚖️ 持有觀望"

# --- 10. AI 與 PDF 輸出模組 ---
def get_valid_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            models = r.json().get('models', [])
            for m in models:
                if 'flash' in m['name']: return m['name'].split('/')[-1]
            for m in models:
                if 'pro' in m['name']: return m['name'].split('/')[-1]
    except: pass
    return "gemini-1.5-flash"

def call_ai(prompt):
    if not api_key: return "⚠️ 尚未設定 API Key"
    if not st.session_state.get('ai_model_name'):
        st.session_state['ai_model_name'] = get_valid_model(api_key)
    target_model = st.session_state['ai_model_name']
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)
        if r.status_code == 200: return r.json()['candidates'][0]['content']['parts'][0]['text']
        else: return f"分析服務暫時無回應 (代碼: {r.status_code})"
    except Exception as e: return "分析服務連線逾時，請稍後再試。"

def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    font_name = font_name_global 
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=20, alignment=1, spaceAfter=20)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName=font_name, fontSize=14, spaceBefore=10, spaceAfter=10, textColor=colors.darkblue)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14)
    
    logic_name = "量化風控模型" if st.session_state['current_logic'] == "Quant" else "價值護城河模型"
    story.append(Paragraph(f"台股量化與價值分析終端 - 綜合洞察報告 [{logic_name}]", title_style))
    story.append(Paragraph(f"產出時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 10))
    
    safe_name = html.escape(str(stock_data.get('名稱', '')))
    safe_code = html.escape(str(stock_data.get('代號', '')))
    story.append(Paragraph(f"分析標的: {safe_name} ({safe_code})", h2_style))
    story.append(Spacer(1, 10))
    
    def safe_str(val, fmt="{:.2f}"):
        try: return "N/A" if (pd.isna(val) or val is None) else fmt.format(float(val))
        except: return "N/A"

    metrics_data = [
        ['綜合評分 (Score)', f"{stock_data.get('Score', 'N/A')}", '收盤價 (Price)', f"{stock_data.get('close_price', 'N/A')}"],
        ['夏普值 (Sharpe)', safe_str(stock_data.get('sharpe')), '波動率 (Volatility)', safe_str(stock_data.get('volatility'), "{:.1f}%")],
        ['Beta (風險係數)', safe_str(stock_data.get('beta'), "{:.2f}"), '最大回撤 (MDD)', safe_str(stock_data.get('mdd')*100 if not pd.isna(stock_data.get('mdd')) else np.nan, "{:.1f}%")],
        ['ROE (權益報酬)', safe_str(stock_data.get('roe'), "{:.1f}%"), '毛利率 (Gross Margin)', safe_str(stock_data.get('gross_margins'), "{:.1f}%")],
        ['營收 YoY (Rev Grw)', safe_str(stock_data.get('rev_growth'), "{:.1f}%"), 'EPS YoY (EPS Grw)', safe_str(stock_data.get('eps_growth'), "{:.1f}%")],
        ['本益比 (P/E)', safe_str(stock_data.get('pe')), 'PEG 估值', safe_str(stock_data.get('peg'))],
        ['殖利率 (Yield)', safe_str(stock_data.get('yield'), "{:.2f}%"), 'FCF 收益率 (FCF Yld)', safe_str(stock_data.get('fcf_yield'), "{:.2f}%")],
        ['負債權益比 (D/E)', safe_str(stock_data.get('de_ratio'), "{:.1f}%"), '隱含成長 (Implied G)', safe_str(stock_data.get('implied_growth')*100 if not pd.isna(stock_data.get('implied_growth')) else np.nan, "{:.1f}%")]
    ]
    t = Table(metrics_data, colWidths=[120, 110, 120, 110])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    if 'ai_analysis' in stock_data and stock_data['ai_analysis']:
        story.append(Paragraph("AI 系統深度洞察", h2_style))
        clean_text = stock_data['ai_analysis']
        for line in clean_text.split('\n'):
            if line.strip():
                safe_line = html.escape(line.strip()).replace('**', '').replace('##', '')
                story.append(Paragraph(safe_line, normal_style))
                story.append(Spacer(1, 5))
    try: doc.build(story)
    except: pass
    buffer.seek(0)
    return buffer

AI_PROMPT_TEMPLATE = """
請扮演專業的法人機構量化研究員，使用**繁體中文 (Traditional Chinese)** 分析 [STOCK] ([SECTOR])。
【時間基準】今天是 [CURRENT_DATE]，請以最新視角撰寫，絕對不要在報告中捏造或顯示過去的歷史日期。

【驅動模型】：[LOGIC_NAME]
[LOGIC_DESC]

【量化與財務數據】
1. 量化風險：夏普值=[SHARPE], 波動率=[VOL]%, Beta=[BETA], 最大回撤=[MDD]%.
2. 價值護城河：ROE=[ROE]%, 毛利率=[GM]%, FCF收益率=[FCF_Y]%, 負債權益比=[DE]%, EPS成長=[EPS_G]%.
3. 估值與時機：PE=[PE], Reverse DCF 隱含成長=[IMPLIED_G]%, 季線乖離=[MA_BIAS]%.

【輸出要求】
請依照「[LOGIC_NAME]」模型視角，輸出三大分析模塊：
1. **風險與效率前緣檢驗**：依據清大林哲群教授《紀律長贏》的學理，嚴格檢視夏普值是否匹配風險，並特別注意「最大回撤(MDD)」是否跌破 25% 防線。解釋此標的在「成長/防禦/中性」三類資產中的戰略定位。
2. **護城河與估值**：檢視財務體質，並對照 Reverse DCF 評估是否透支未來。
3. **操作結論**：結合 MDD 與季線乖離，給出客觀的資產配置定位與具體行動建議。
"""

# --- 15. 主介面 ---
with st.sidebar:
    st.title("⚙️ 終端控制面板")
    logic_choice_tw = st.radio("選擇運算引擎", ["📊 量化風控模型 (夏普值/波動率/Beta)", "👴 價值護城河 (高ROE/毛利/現金流)"], label_visibility="collapsed")
    st.session_state['current_logic'] = "Quant" if "量化" in logic_choice_tw else "Buffett"
    st.markdown("---")
    
    scan_mode = st.radio("篩選維度", ["市場焦點策略", "產業族群板塊", "台灣 ETF 專區", "自訂代碼輸入"])
    target_stocks = []
    
    strategies = {
        "動能爆發 (高夏普)": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "3661.TW"],
        "低波動防禦 (低標準差)": ["2412.TW", "3045.TW", "2881.TW", "2892.TW", "1101.TW"],
        "蘋果與消費電 (Beta循環)": ["2330.TW", "2317.TW", "3008.TW", "4938.TW", "2313.TW"],
        "電動車與車電 (趨勢Beta)": ["2308.TW", "2317.TW", "6235.TW", "1536.TW", "5425.TW"]
    } if st.session_state['current_logic'] == "Quant" else {
        "台灣50 (護城河權值)": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2412.TW", "1301.TW"],
        "高股息與穩健 (現金流)": ["2454.TW", "2303.TW", "2357.TW", "1101.TW", "2891.TW"],
        "金融保險 (防禦收息)": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "5880.TW", "2884.TW"],
        "傳產與航運 (循環價值)": ["2603.TW", "2609.TW", "2002.TW", "1301.TW", "1303.TW", "1605.TW"]
    }
        
    etf_strategies = {
        "市值型 ETF (大盤連動)": ["0050.TW", "006208.TW", "00692.TW", "00881.TW"],
        "高股息 ETF (穩定配息)": ["0056.TW", "00878.TW", "00919.TW", "00929.TW", "00713.TW"],
        "科技與半導體主題": ["00891.TW", "00892.TW", "00881.TW", "00830.TW"],
        "海外與美股連結": ["00757.TW", "00646.TW", "00830.TW", "00662.TW"]
    }
        
    if scan_mode == "自訂代碼輸入":
        target_stocks = st.multiselect("搜尋上市櫃標的", list(stock_map.values()), default=["2330.TW 台積電", "2454.TW 聯發科"])
        if manual := st.text_input("快速輸入代號 (如 2317)"): target_stocks.append(f"{manual}.TW")
    elif scan_mode == "產業族群板塊":
        selected_inds = st.multiselect("板塊選擇", ["[ALL] 載入全部板塊"] + sorted(list(industry_map.keys())))
        for k in (industry_map.keys() if "[ALL] 載入全部板塊" in selected_inds else selected_inds): target_stocks.extend(industry_map[k])
    elif scan_mode == "台灣 ETF 專區":
        etf_keys = list(etf_strategies.keys())
        selected_etfs = st.multiselect("ETF 分類 (支援複選)", ["[ALL] 載入全部 ETF"] + etf_keys, default=[etf_keys[0]])
        for k in (etf_keys if "[ALL] 載入全部 ETF" in selected_etfs else selected_etfs): target_stocks.extend(etf_strategies[k])
    else: 
        strat_keys = list(strategies.keys())
        selected_strats = st.multiselect("推薦策略池", ["[ALL] 載入全部策略"] + strat_keys, default=[strat_keys[0]])
        for k in (strat_keys if "[ALL] 載入全部策略" in selected_strats else selected_strats): target_stocks.extend(strategies[k])

    target_stocks = list(dict.fromkeys(target_stocks)) 

    if st.button("🚀 啟動終端運算 (強制擷取最新數據)", type="primary", use_container_width=True):
        st.session_state['scan_finished'] = False
        st.session_state['panel_page'] = 1 
        st.session_state['list_page'] = 1 
        current_date_str = datetime.now().strftime("%Y/%m/%d")
        with st.spinner(f"正在全域聚合 {current_date_str} 盤中最新財報與量化數據 (共 {len(target_stocks)} 檔)..."):
            raw, hist_map = batch_scan_stocks(target_stocks)
            
            if not raw.empty:
                st.session_state['raw_data'] = sanitize_data(raw)
                st.session_state['history_storage'] = hist_map
                st.session_state['scan_finished'] = True
            else:
                st.error("❌ 掃描失敗：請檢查網路連線或該標的代碼是否正確。")
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 台股量化與價值分析終端")
    st.caption(f"STATUS: ONLINE | ENGINE: **{'量化風控' if st.session_state['current_logic'] == 'Quant' else '價值護城河'}** | ALGO: 防呆自研推導引擎滿血版")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    
    if df.empty: st.error("檢索結果為空，請確認代碼有效性。")
    else:
        final_df, df_norm = calculate_score(df, logic_type=st.session_state['current_logic'])
        
        st.subheader("🏆 終端檢索清單")
        tags = sorted(final_df['Quality'].dropna().unique().tolist(), key=lambda t: 0 if "成長" in t else (1 if "防禦" in t else (2 if "中性" in t else 3)))
        selected_filter = st.radio("🎯 快速分類篩選：", ["[ALL] 顯示全部標的"] + tags, horizontal=True)
        filtered_df = final_df if selected_filter == "[ALL] 顯示全部標的" else final_df[final_df['Quality'] == selected_filter]
        
        ROWS_PER_PAGE = 50
        total_list_pages = max(1, (len(filtered_df) - 1) // ROWS_PER_PAGE + 1)
        if st.session_state['list_page'] > total_list_pages:
            st.session_state['list_page'] = total_list_pages

        list_start_idx = (st.session_state['list_page'] - 1) * ROWS_PER_PAGE
        list_end_idx = list_start_idx + ROWS_PER_PAGE
        paged_filtered_df = filtered_df.iloc[list_start_idx:list_end_idx]

        df_event = st.dataframe(
            paged_filtered_df[['代號', '名稱', 'industry', 'Score', 'Quality', 'Strategy', 'sharpe', 'mdd', 'roe', 'implied_growth']],
            column_config={
                "Score": st.column_config.ProgressColumn("綜合評分", min_value=0, max_value=100, format="%.1f"),
                "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"),
                "mdd": st.column_config.NumberColumn("最大回撤", format="%.1f%%"),
                "roe": st.column_config.NumberColumn("ROE", format="%.1f"),
            },
            use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
        )

        c_lp1, c_lp2, c_lp3 = st.columns([1, 2, 1])
        if c_lp1.button("⬅️ 上一頁 (清單)", use_container_width=True, disabled=st.session_state['list_page']==1):
            st.session_state['list_page'] -= 1
            st.rerun()
        c_lp2.markdown(f"<div style='text-align:center;color:#9CA3AF;'>清單第 <b>{st.session_state['list_page']}</b> 頁 / 共 <b>{total_list_pages}</b> 頁 (總計 {len(filtered_df)} 檔)</div>", unsafe_allow_html=True)
        if c_lp3.button("下一頁 ➡️ (清單)", use_container_width=True, disabled=st.session_state['list_page']==total_list_pages):
            st.session_state['list_page'] += 1
            st.rerun()
        st.markdown("---")

        selected_rows = df_event.selection.rows
        
        if selected_rows:
            sel_idx = list_start_idx + selected_rows[0]
            sel_row = filtered_df.iloc[sel_idx]
            c_title, c_add = st.columns([3, 1])
            c_title.subheader(f"💡 系統標籤深度解析：{sel_row['名稱']} ({sel_row['代號']})")
            if c_add.button("❌ 從自選移除" if sel_row['代號'] in st.session_state['my_portfolio'] else "➕ 加入自選戰略組合", use_container_width=True):
                st.session_state['my_portfolio'].remove(sel_row['代號']) if sel_row['代號'] in st.session_state['my_portfolio'] else st.session_state['my_portfolio'].append(sel_row['代號'])
                st.rerun()
            st.markdown(f"<div class='explain-box'>{get_tag_explanation(sel_row, st.session_state['current_logic'])}</div>", unsafe_allow_html=True)
            st.subheader("🎯 個股深度檢驗面板 (單檔檢視)")
            display_df = filtered_df.iloc[[sel_idx]]
        else:
            st.subheader("🎯 個股深度檢驗面板 (全覽分頁)")
            total_pages = max(1, (len(filtered_df) - 1) // 5 + 1) if len(filtered_df) > 0 else 1
            if st.session_state['panel_page'] > total_pages: st.session_state['panel_page'] = total_pages
            start_idx = (st.session_state['panel_page'] - 1) * 5
            display_df = filtered_df.iloc[start_idx:start_idx+5]
            
        def n2s(v, fmt): return "N/A" if pd.isna(v) else fmt.format(v)

        for idx, row in display_df.iterrows():
            code = row['代號']
            q_tag = row['Quality']
            tag_html = f"<span class='tag tag-moat'>💎 {q_tag}</span>" if "成長" in q_tag or "防禦" in q_tag or "護城河" in q_tag else (f"<span class='tag tag-logic'>⚖️ {q_tag}</span>" if "中性" in q_tag else (f"<span class='tag tag-risk'>⚠️ {q_tag}</span>" if "風險" in q_tag or "剔除" in q_tag else ""))
            
            st.markdown(f"""<div class='stock-card'>
<div class='card-header'>
<div><span class='header-title'>{row['名稱']} ({code})</span><span class='header-price'>${row['close_price']}</span></div>
<div><span class='tag tag-logic'>{'量化風控' if st.session_state['current_logic'] == 'Quant' else '价值護城河'}</span><span class='tag tag-sector'>{row['industry']}</span>{tag_html}</div>
</div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1, 1.8, 1.5])
            with c1:
                orig_idx = df_norm.index[df_norm['代號'] == code]
                if len(orig_idx) > 0: st.plotly_chart(plot_radar_chart_ui(row['名稱'], get_radar_data(df_norm.loc[orig_idx[0]])), use_container_width=True, key=f"radar_{code}")
            with c2:
                sh = row.get('sharpe')
                mdd = row.get('mdd')
                rev = row.get('rev_growth')
                eps = row.get('eps_growth')
                
                sh_c = "" if pd.isna(sh) or sh == 0 else ("q-up" if sh>1 else ("q-down" if sh<0 else ""))
                mdd_c = "" if pd.isna(mdd) else ("q-up" if mdd>-0.15 else ("q-down" if mdd<-0.25 else "q-neu"))
                rev_c = "" if pd.isna(rev) or rev == 0 else ("q-up" if rev>0 else "q-down")
                eps_c = "" if pd.isna(eps) or eps == 0 else ("q-up" if eps>0 else "q-down")

                st.markdown(f"""<div class='quote-board'>
<div class='quote-item'><span class='q-label'>夏普值 (Sharpe)</span><span class='q-val {sh_c}'>{n2s(sh, "{:.2f}")}</span></div>
<div class='quote-item'><span class='q-label'>波動率 (Volatility)</span><span class='q-val'>{n2s(row.get('volatility')*100 if pd.notna(row.get('volatility')) else np.nan, "{:.1f}%")}</span></div>
<div class='quote-item'><span class='q-label'>Beta (風險係數)</span><span class='q-val'>{n2s(row.get('beta'), "{:.2f}")}</span></div>
<div class='quote-item'><span class='q-label'>最大回撤 (MDD)</span><span class='q-val {mdd_c}'>{n2s(mdd*100 if pd.notna(mdd) else np.nan, "{:.1f}%")}</span></div>
<div class='quote-item'><span class='q-label'>ROE (權益報酬)</span><span class='q-val'>{n2s(row.get('roe'), "{:.1f}%")}</span></div>
<div class='quote-item'><span class='q-label'>毛利率 (Gross Mgn)</span><span class='q-val'>{n2s(row.get('gross_margins'), "{:.1f}%")}</span></div>
<div class='quote-item'><span class='q-label'>營收 YoY</span><span class='q-val {rev_c}'>{n2s(rev, "{:.1f}%")}</span></div>
<div class='quote-item'><span class='q-label'>EPS YoY</span><span class='q-val {eps_c}'>{n2s(eps, "{:.1f}%")}</span></div>
<div class='quote-item'><span class='q-label'>殖利率 (Yield)</span><span class='q-val'>{n2s(row.get('yield'), "{:.2f}%")}</span></div>
<div class='quote-item'><span class='q-label'>本益比 (P/E)</span><span class='q-val'>{n2s(row.get('pe'), "{:.2f}")}</span></div>
<div class='quote-item'><span class='q-label'>PEG 估值</span><span class='q-val'>{n2s(row.get('peg'), "{:.2f}")}</span></div>
<div class='quote-item'><span class='q-label'>FCF 收益率</span><span class='q-val'>{n2s(row.get('fcf_yield'), "{:.2f}%")}</span></div>
</div>""", unsafe_allow_html=True)
                
                ig = row.get('implied_growth', np.nan)
                if not pd.isna(ig):
                    status = "<span class='q-down'>🔴 預估過熱</span>" if ig > (eps if pd.notna(eps) else 0)/100 + 0.1 else ("<span class='q-up'>🟢 具安全邊際</span>" if ig < (eps if pd.notna(eps) else 0)/100 - 0.05 else "<span class='q-neu'>🟡 估值公允</span>")
                    st.markdown(f"<div class='dcf-panel'><div style='font-size:0.8rem;color:#9CA3AF;'>REVERSE DCF 估值模型</div><div style='display:flex;justify-content:space-between;'><span style='color:#F8FAFC;'>市場隱含期望: <b style='font-size:1.2rem;'>{ig*100:.1f}%</b></span><span>{status}</span></div></div>", unsafe_allow_html=True)
                
                c_btn1, c_btn2 = st.columns(2)
                if code not in st.session_state['ai_results']:
                    if c_btn1.button(f"🧠 執行 AI 洞察", key=f"ai_{code}"):
                        p_txt = AI_PROMPT_TEMPLATE.replace("[LOGIC_NAME]", "量化風控模型" if st.session_state['current_logic'] == "Quant" else "價值護城河模型") \
                            .replace("[LOGIC_DESC]", "評估最大回撤防禦極限" if st.session_state['current_logic'] == "Quant" else "優先考量ROE現金流") \
                            .replace("[STOCK]", row['名稱']).replace("[SECTOR]", str(row['industry'])) \
                            .replace("[CURRENT_DATE]", datetime.now().strftime('%Y年%m月%d日')) \
                            .replace("[SHARPE]", n2s(sh, "{:.2f}")).replace("[VOL]", n2s(row.get('volatility')*100 if pd.notna(row.get('volatility')) else np.nan, "{:.1f}")) \
                            .replace("[BETA]", n2s(row.get('beta'), "{:.2f}")).replace("[MDD]", n2s(mdd*100 if pd.notna(mdd) else np.nan, "{:.1f}")) \
                            .replace("[ROE]", n2s(row.get('roe'), "{:.1f}")).replace("[GM]", n2s(row.get('gross_margins'), "{:.1f}")) \
                            .replace("[FCF_Y]", n2s(row.get('fcf_yield'), "{:.2f}")).replace("[DE]", n2s(row.get('de_ratio'), "{:.1f}")) \
                            .replace("[EPS_G]", n2s(eps, "{:.1f}")).replace("[PE]", n2s(row.get('pe'), "{:.2f}")) \
                            .replace("[IMPLIED_G]", n2s(ig*100 if pd.notna(ig) else np.nan, "{:.1f}")).replace("[MA_BIAS]", n2s(row.get('priceToMA60')*100 if pd.notna(row.get('priceToMA60')) else np.nan, "{:.1f}"))
                        st.session_state['ai_results'][code] = call_ai(p_txt)
                        st.rerun() 
                else:
                    if c_btn1.button(f"🔄 更新 AI 洞察", key=f"ai_re_{code}"):
                        p_txt = AI_PROMPT_TEMPLATE.replace("[LOGIC_NAME]", "量化風控模型" if st.session_state['current_logic'] == "Quant" else "價值護城河模型") \
                            .replace("[LOGIC_DESC]", "評估最大回撤防禦極限" if st.session_state['current_logic'] == "Quant" else "優先考量ROE現金流") \
                            .replace("[STOCK]", row['名稱']).replace("[SECTOR]", str(row['industry'])) \
                            .replace("[CURRENT_DATE]", datetime.now().strftime('%Y年%m月%d日')) \
                            .replace("[SHARPE]", n2s(sh, "{:.2f}")).replace("[VOL]", n2s(row.get('volatility')*100 if pd.notna(row.get('volatility')) else np.nan, "{:.1f}")) \
                            .replace("[BETA]", n2s(row.get('beta'), "{:.2f}")).replace("[MDD]", n2s(mdd*100 if pd.notna(mdd) else np.nan, "{:.1f}")) \
                            .replace("[ROE]", n2s(row.get('roe'), "{:.1f}")).replace("[GM]", n2s(row.get('gross_margins'), "{:.1f}")) \
                            .replace("[FCF_Y]", n2s(row.get('fcf_yield'), "{:.2f}")).replace("[DE]", n2s(row.get('de_ratio'), "{:.1f}")) \
                            .replace("[EPS_G]", n2s(eps, "{:.1f}")).replace("[PE]", n2s(row.get('pe'), "{:.2f}")) \
                            .replace("[IMPLIED_G]", n2s(ig*100 if pd.notna(ig) else np.nan, "{:.1f}")).replace("[MA_BIAS]", n2s(row.get('priceToMA60')*100 if pd.notna(row.get('priceToMA60')) else np.nan, "{:.1f}"))
                        st.session_state['ai_results'][code] = call_ai(p_txt)
                        st.rerun()
                    
                    pdf_payload = row.to_dict()
                    pdf_payload['ai_analysis'] = st.session_state['ai_results'][code]
                    c_btn2.download_button("📥 輸出 PDF 報告", create_pdf(pdf_payload), f"{code} {row['名稱']}_Report.pdf", key=f"dl_{code}")

            with c3:
                h_df = hist_storage.get(code)
                if h_df is not None and not h_df.empty: st.plotly_chart(plot_trend_dashboard(row['名稱'], h_df, row.get('priceToMA60', 0)), use_container_width=True, key=f"trend_{code}")
                else: st.warning("無 K 線數據")
                
            if code in st.session_state['ai_results']:
                st.markdown(f"<div class='ai-box'>{st.session_state['ai_results'][code]}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if not selected_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            cp1, cp2, cp3 = st.columns([1, 2, 1])
            if cp1.button("⬅️ 上一頁", use_container_width=True, disabled=st.session_state['panel_page']==1): st.session_state['panel_page']-=1; st.rerun()
            cp2.markdown(f"<div style='text-align:center;color:#9CA3AF;'>第 <b>{st.session_state['panel_page']}</b> 頁 / 共 <b>{total_pages}</b> 頁</div>", unsafe_allow_html=True)
            if cp3.button("下一頁 ➡️", use_container_width=True, disabled=st.session_state['panel_page']==total_pages): st.session_state['panel_page']+=1; st.rerun()

        # =========================================================
        # 🛠️ 戰略指揮中心：自選組合監控與多空情境推演
        # =========================================================
        st.markdown("---")
        st.subheader("🛠️ 戰略指揮中心：自選組合監控與多空情境推演")
        regime = st.radio("🌍 切換總經市場情境", ["📈 多頭市場 (Bull Market) - 放大獲利", "📉 空頭/震盪市場 (Bear Market) - 著重防禦"], horizontal=True)
        
        if not st.session_state['my_portfolio']:
            st.info("💡 請在上方清單點選標的，並點擊【➕ 加入自選戰略組合】以啟用監控與推演功能。")
        else:
            my_port_df = final_df[final_df['代號'].isin(st.session_state['my_portfolio'])].copy()
            
            my_port_df['資產屬性'] = my_port_df.apply(lambda r: "🛑 剔除資產" if r.get('mdd',0) < -0.25 else ("🚀 成長型資產" if r.get('sharpe',0)>1 and r.get('beta',1)>=1 else ("🛡️ 防禦型資產" if r.get('sharpe',0)>1 and r.get('beta',1)<1 else ("⚖️ 中性資產" if r.get('sharpe',0)>0 and r.get('volatility',1)<25 else "🟡 觀察資產"))), axis=1)
            my_port_df['操作訊號'] = my_port_df.apply(lambda r: generate_custom_signal(r, "Bull" if "多頭" in regime else "Bear"), axis=1)
            
            st.dataframe(my_port_df[['代號', '名稱', '資產屬性', 'sharpe', 'beta', 'mdd', '操作訊號']].sort_values('sharpe', ascending=False), hide_index=True, use_container_width=True)
            
            counts = my_port_df['資產屬性'].value_counts()
            if "多頭" in regime: t_grow, t_neu, t_def, t_cash, adv = 60, 25, 10, 5, "提高成長型資產配置至 60%，放大高 Sharpe 動能標的權重。"
            else: t_grow, t_neu, t_def, t_cash, adv = 30, 30, 25, 15, "防禦優先，降低高 Beta 持倉，將防禦與中性資產拉高至 55%，並保留充裕現金。"
            
            st.markdown(f"<div class='cmd-box'><b>📝 情境推演建議：</b><br>{adv}</div>", unsafe_allow_html=True)
            w_c1, w_c2, w_c3, w_c4 = st.columns(4)
            for col, title, tgt, count in zip([w_c1, w_c2, w_c3, w_c4], ["🚀 成長型資產", "⚖️ 中性資產", "🛡️ 防禦型資產", "💵 現金部位"], [t_grow, t_neu, t_def, t_cash], [counts.get("🚀 成長型資產",0), counts.get("⚖️ 中性資產",0), counts.get("🛡️ 防禦型資產",0), "-"]):
                col.markdown(f"<div style='background-color:#06090F; padding:15px; border-radius:4px; border:1px solid #1E293B;'><div style='color:#9CA3AF;'>{title}</div><div style='font-size:1.5rem; font-weight:bold; color:#F8FAFC;'>目標 {tgt}%</div><div style='color:#3B82F6;'>目前自選: {count}</div></div>", unsafe_allow_html=True)

        # =========================================================
        # 💼 系統嚴選：模型專屬效率前緣配置
        # =========================================================
        st.markdown("---")
        st.subheader("💼 系統嚴選：模型專屬配置 (依總經多空動態調整)")
        
        df_list = []
        for c, h_df in hist_storage.items():
            if h_df is not None and not h_df.empty and 'Close' in h_df.columns:
                df_list.append(h_df['Close'].pct_change().rename(c))
                
        if df_list:
            returns_df = pd.concat(df_list, axis=1).dropna(how='all').fillna(0)
            cov_matrix = returns_df.cov() * 252 
        else:
            returns_df = pd.DataFrame()
            cov_matrix = pd.DataFrame()
        
        if st.session_state['current_logic'] == "Quant":
            pool_df = final_df[(final_df['sharpe'] > 0) & (final_df['mdd'] >= -0.25)].copy()
            if len(pool_df)==0: pool_df = final_df.head(10).copy()
            pool_df['戰略定位'] = pool_df.apply(lambda r: "🚀 成長型資產" if r.get('sharpe',0)>1 and r.get('beta',1)>=1 else ("🛡️ 防禦型資產" if r.get('sharpe',0)>1 and r.get('beta',1)<1 else "⚖️ 中性資產"), axis=1)
            
            is_bull = "多頭" in regime
            target_g = 5 if is_bull else 2
            target_d = 2 if is_bull else 4
            target_n = 3 if is_bull else 4
            
            f_codes = greedy_mpt_optimization(pool_df[pool_df['戰略定位']=="🚀 成長型資產"], returns_df, target_g, 'sharpe', target_vol_max=0.15)
            f_codes = greedy_mpt_optimization(pool_df[pool_df['戰略定位']=="🛡️ 防禦型資產"], returns_df, target_d, 'sharpe', initial_selected=f_codes)
            f_codes = greedy_mpt_optimization(pool_df[pool_df['戰略定位']=="⚖️ 中性資產"], returns_df, target_n, 'sharpe', initial_selected=f_codes)
            if len(f_codes)<10: f_codes.extend(greedy_mpt_optimization(pool_df[~pool_df['代號'].isin(f_codes)], returns_df, 10-len(f_codes), 'sharpe', initial_selected=f_codes))
            
            port_df = pool_df[pool_df['代號'].isin(f_codes)].copy()
            
            def get_weight_text(row):
                t = row['戰略定位']
                if is_bull: return "10%~15%" if t == "🚀 成長型資產" else ("5%~8%" if t == "🛡️ 防禦型資產" else "8%~10%")
                else: return "5%~8%" if t == "🚀 成長型資產" else ("12%~15%" if t == "🛡️ 防禦型資產" else "8%~12%")
            
            port_df['建議權重'] = port_df.apply(get_weight_text, axis=1)
            
            c1, c2, c3 = st.columns([1.35, 0.9, 1.25])
            with c1:
                st.dataframe(port_df[['代號', '名稱', '戰略定位', 'sharpe', 'mdd', '建議權重']], hide_index=True, use_container_width=True)
            with c2:
                port_codes = port_df['代號'].tolist()
                true_vol = 0
                avg_sharpe = port_df['sharpe'].mean()
                avg_vol = port_df['volatility'].mean() * 100
                avg_beta = port_df['beta'].mean()
                
                if not cov_matrix.empty and len(port_codes) > 1:
                    valid_codes = [c for c in port_codes if c in cov_matrix.columns]
                    if len(valid_codes) == len(port_codes):
                        def get_num_weight(row):
                            t = row['戰略定位']
                            if is_bull: return 0.12 if t == "🚀 成長型資產" else (0.05 if t == "🛡️ 防禦型資產" else 0.10)
                            else: return 0.05 if t == "🚀 成長型資產" else (0.125 if t == "🛡️ 防禦型資產" else 0.10)
                        
                        raw_w = np.array([get_num_weight(r) for _, r in port_df.iterrows()])
                        weights = raw_w / np.sum(raw_w) if np.sum(raw_w) > 0 else np.ones(len(raw_w))/len(raw_w)
                        true_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.loc[valid_codes, valid_codes], weights))) * 100
                
                vol_color = "#10B981" if true_vol <= 15 else "#EF4444"
                st.markdown(f"""
                <div style='background-color:#1E293B; padding:15px; border-radius:4px; border:1px solid #334155;'>
                    <h4 style='margin-top:0; color:#F8FAFC; border-bottom:1px solid #334155; padding-bottom:10px;'>MPT 效率前緣檢驗</h4>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#9CA3AF;'>單檔平均波動率</span><span style='color:#F8FAFC; font-weight:bold; font-size:1.1rem;'>{avg_vol:.1f}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#F8FAFC; font-weight:600;'>動態權重組合波動率</span><span style='color:{vol_color}; font-weight:bold; font-size:1.3rem;'>{true_vol:.1f}%</span>
                    </div>
                    <div style='font-size:0.8rem; color:#64748B; margin-bottom:12px;'>* 已依多空動態權重加權共變異數</div>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#9CA3AF;'>組合總體 Beta</span><span style='color:#F8FAFC; font-weight:bold; font-size:1.1rem;'>{avg_beta:.2f}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#9CA3AF;'>組合平均夏普值</span><span style='color:#10B981; font-weight:bold; font-size:1.1rem;'>{avg_sharpe:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                if not returns_df.empty and len(port_codes) > 1:
                    v_codes = [c for c in port_codes if c in returns_df.columns]
                    if len(v_codes) > 1:
                        corr_mat = returns_df[v_codes].corr()
                        corr_mat.columns = [str(c).split('.')[0] for c in v_codes]
                        corr_mat.index = [str(c).split('.')[0] for c in v_codes]
                        fig_corr = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="組合資產關聯熱力圖")
                        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9CA3AF', size=11), margin=dict(t=40, b=20, l=20, r=20), height=380)
                        fig_corr.update_xaxes(type='category')
                        fig_corr.update_yaxes(type='category')
                        st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.caption("依據巴菲特價值投資哲學：MDD < -25% 無情剔除！嚴選具備「高 ROE、高毛利、充沛自由現金流」的優質資產。系統將依據公司 ROE 的強弱，自動進行資金的集中最佳化權重分配。")
            pool_df = final_df[(final_df['roe'] > 15) & (final_df['gross_margins'] > 40) & (final_df['mdd'] >= -0.25)].copy()
            if len(pool_df)<10: pool_df = final_df[final_df['mdd'] >= -0.25].nlargest(10, 'Score').copy()
            pool_df['戰略定位'] = pool_df.apply(lambda r: "🚀 成長護城河" if r.get('eps_growth',0)>15 else "💰 穩健價值", axis=1)
            
            is_bull = "多頭" in regime
            target_g = 6 if is_bull else 3
            target_v = 4 if is_bull else 7
            
            pool_g = pool_df[pool_df['戰略定位'] == "🚀 成長護城河"].sort_values('Score', ascending=False).head(target_g)
            pool_v = pool_df[pool_df['戰略定位'] == "💰 穩健價值"].sort_values('Score', ascending=False).head(target_v)
            port_df = pd.concat([pool_g, pool_v])
            if len(port_df) < 10:
                rem = pool_df[~pool_df['代號'].isin(port_df['代號'])].sort_values('Score', ascending=False).head(10 - len(port_df))
                port_df = pd.concat([port_df, rem])
            
            valid_roe = port_df['roe'].apply(lambda x: x if pd.notna(x) and x > 0 else 0.1)
            if not is_bull:
                valid_roe = valid_roe * port_df['戰略定位'].apply(lambda x: 1.5 if x == "💰 穩健價值" else 1.0)
            
            total_wt = valid_roe.sum()
            port_df['建議權重'] = (valid_roe / total_wt * 100).apply(lambda x: f"{x:.1f}%") if total_wt > 0 else "10.0%"
            
            c1, c2, c3 = st.columns([1.35, 0.9, 1.25])
            with c1:
                st.dataframe(port_df[['代號', '名稱', '戰略定位', 'roe', 'gross_margins', 'pe', '建議權重']], hide_index=True, use_container_width=True)
            with c2:
                avg_roe = port_df['roe'].mean()
                avg_gm = port_df['gross_margins'].mean()
                avg_fcf = port_df['fcf_yield'].mean()
                avg_pe = port_df['pe'].mean()
                st.markdown(f"""
                <div style='background-color:#1E293B; padding:15px; border-radius:4px; border:1px solid #334155;'>
                    <h4 style='margin-top:0; color:#F8FAFC; border-bottom:1px solid #334155; padding-bottom:10px;'>護城河總體檢驗 (Moat Quality)</h4>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#9CA3AF;'>組合平均 ROE</span><span style='color:#10B981; font-weight:bold; font-size:1.1rem;'>{avg_roe:.1f}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#9CA3AF;'>組合平均毛利率</span><span style='color:#10B981; font-weight:bold; font-size:1.1rem;'>{avg_gm:.1f}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
                        <span style='color:#9CA3AF;'>組合 FCF 收益率</span><span style='color:#F8FAFC; font-weight:bold; font-size:1.1rem;'>{avg_fcf:.2f}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#9CA3AF;'>組合平均本益比</span><span style='color:#F8FAFC; font-weight:bold; font-size:1.1rem;'>{avg_pe:.1f}x</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                moat_df = port_df[['代號', '名稱', 'roe', 'gross_margins']].copy()
                moat_df['short_name'] = moat_df['代號'].astype(str).str.replace('.TW', '').str.replace('.TWO', '')
                
                fig_bar = go.Figure(data=[
                    go.Bar(name='ROE (%)', x=moat_df['short_name'], y=moat_df['roe'], marker_color='#10B981'),
                    go.Bar(name='毛利率 (%)', x=moat_df['short_name'], y=moat_df['gross_margins'], marker_color='#3B82F6')
                ])
                fig_bar.update_layout(
                    title="組合企業護城河指標 (Quality Metrics)",
                    barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='#9CA3AF', size=11), margin=dict(t=40, b=20, l=20, r=20), height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_bar.update_xaxes(type='category', showgrid=False, linecolor='#334155')
                fig_bar.update_yaxes(showgrid=True, gridcolor='#1E293B')
                st.plotly_chart(fig_bar, use_container_width=True)
