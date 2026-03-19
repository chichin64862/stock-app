import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import io
import os
import time
from datetime import datetime
import urllib.request
import html

# --- 關閉 SSL 驗證警告 (針對證交所官方 API) ---
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
st.set_page_config(
    page_title="台股量化與價值分析終端", 
    page_icon="Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 華爾街專業看盤軟體風格 ---
st.markdown("""
<style>
    :root { color-scheme: dark !important; }
    .stApp { background-color: #06090F !important; }
    [data-testid="stSidebar"] { background-color: #0D131F !important; border-right: 1px solid #1E293B !important; }
    h1, h2, h3, p, span, div, label { font-family: 'Segoe UI', Tahoma, sans-serif; color: #E2E8F0 !important; }
    
    .stMultiSelect [data-baseweb="select"] { background-color: #0D131F !important; }
    .stMultiSelect [data-baseweb="select"] > div { background-color: #0D131F !important; color: #F8FAFC !important; border-color: #1E293B !important; }
    .stMultiSelect span[data-baseweb="tag"] { background-color: #1E293B !important; color: #F8FAFC !important; }
    .stMultiSelect span[data-baseweb="tag"] span { color: #F8FAFC !important; }
    .stSelectbox [data-baseweb="select"] > div { background-color: #0D131F !important; color: #F8FAFC !important; border-color: #1E293B !important; }
    ul[role="listbox"] { background-color: #0D131F !important; color: #F8FAFC !important; border: 1px solid #1E293B !important; }
    li[role="option"] { background-color: #0D131F !important; color: #F8FAFC !important; }
    li[role="option"]:hover { background-color: #3B82F6 !important; }
    input { background-color: #0D131F !important; color: white !important; border: 1px solid #1E293B !important; }
    
    .stock-card { 
        background-color: #0D131F !important; 
        padding: 16px 24px; border-radius: 4px; 
        border: 1px solid #1E293B !important; border-left: 4px solid #3B82F6 !important; 
        margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .card-header {
        display: flex; justify-content: space-between; align-items: flex-end;
        border-bottom: 1px solid #1E293B !important; padding-bottom: 12px; margin-bottom: 16px;
    }
    .header-title { font-size: 1.4rem; font-weight: 700; color: #FFFFFF !important; letter-spacing: 1px; }
    .header-price { font-size: 1.4rem; font-weight: 700; color: #10B981 !important; margin-left: 16px; font-family: 'Consolas', monospace; }
    
    .quote-board {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
        background-color: #1E293B !important; border: 1px solid #1E293B !important; border-radius: 4px;
    }
    .quote-item { 
        background-color: #06090F !important; padding: 12px 16px; display: flex; flex-direction: column; justify-content: center;
    }
    .q-label { color: #64748B !important; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
    .q-val { color: #F8FAFC !important; font-weight: 700; font-size: 1.15rem; font-family: 'Consolas', 'Courier New', monospace; }
    
    .q-up { color: #10B981 !important; }   
    .q-down { color: #EF4444 !important; } 
    .q-neu { color: #F59E0B !important; }  
    
    .tag { padding: 3px 8px; border-radius: 2px; font-size: 0.75rem; font-weight: 700; margin-left: 8px; text-transform: uppercase; border: 1px solid; }
    .tag-moat { background-color: rgba(16, 185, 129, 0.1) !important; color: #10B981 !important; border-color: #10B981 !important; }
    .tag-risk { background-color: rgba(239, 68, 68, 0.1) !important; color: #EF4444 !important; border-color: #EF4444 !important; }
    .tag-logic { background-color: rgba(59, 130, 246, 0.1) !important; color: #3B82F6 !important; border-color: #3B82F6 !important; }

    .dcf-panel {
        background-color: #0D131F !important; border: 1px solid #334155 !important; border-left: 3px solid #8B5CF6 !important;
        padding: 12px 16px; margin-top: 16px; border-radius: 2px;
    }
    
    .ai-box {
        background-color: #0D131F !important; border: 1px solid #1E293B !important; border-left: 3px solid #3B82F6 !important;
        padding: 16px; margin-top: 16px; border-radius: 2px; font-size: 0.95rem; line-height: 1.6; color: #CBD5E1 !important;
    }
    
    .explain-box {
        background-color: #0F172A !important; border: 1px solid #334155 !important; border-left: 4px solid #F59E0B !important;
        padding: 16px 20px; border-radius: 4px; margin-bottom: 24px; color: #E2E8F0 !important; line-height: 1.7; font-size: 0.95rem;
    }
    
    .cmd-box {
        background-color: #1E293B !important; border: 1px solid #334155 !important; border-top: 4px solid #F59E0B !important;
        padding: 20px; border-radius: 4px; margin-bottom: 24px;
    }
    
    .stButton button, .stDownloadButton button { 
        background-color: #1E293B !important; border: 1px solid #334155 !important; 
        color: #F8FAFC !important; border-radius: 2px !important; font-weight: 600 !important;
    }
    .stButton button:hover, .stDownloadButton button:hover { 
        border-color: #3B82F6 !important; color: #3B82F6 !important; background-color: #0D131F !important;
    }
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
if 'my_portfolio' not in st.session_state: st.session_state['my_portfolio'] = []

# --- 4. API Key ---
try: api_key = st.secrets["GEMINI_API_KEY"]
except Exception: st.error("系統偵測不到 API Key，AI 洞察功能將受限。")

# --- 5. 字型下載與註冊 ---
@st.cache_resource
def setup_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    if os.path.exists(font_path) and os.path.getsize(font_path) < 1000000:
        try: os.remove(font_path)
        except: pass
    if not os.path.exists(font_path):
        urls = [
            "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC-Regular.ttf",
            "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.ttf"
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response, open(font_path, 'wb') as out_file:
                    out_file.write(response.read())
                if os.path.getsize(font_path) > 1000000: break
            except Exception: continue
    try:
        if os.path.exists(font_path) and os.path.getsize(font_path) > 1000000:
            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
            return 'ChineseFont'
    except Exception: pass
    try:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        return 'STSong-Light'
    except:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont('MSung-Light'))
            return 'MSung-Light'
        except: return 'Helvetica'

font_name_global = setup_chinese_font()

# --- 6. 核心數據引擎 (多源聚合與防阻擋) ---
def create_resilient_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    return session

# 【內建高質量產業板塊庫，永不漏股】
@st.cache_data(ttl=86400)
def get_tw_stock_list():
    try:
        import twstock
        codes = twstock.codes
        stock_map = {}
        industry_map = {}
        code_to_industry = {}
        
        custom_sector_override = {
            "2330": "半導體業", "2454": "半導體業", "2303": "半導體業", "3711": "半導體業", "3034": "半導體業", "2379": "半導體業",
            "2382": "電腦及週邊設備業", "3231": "電腦及週邊設備業", "2356": "電腦及週邊設備業", "2376": "電腦及週邊設備業", "6669": "電腦及週邊設備業",
            "2308": "電子零組件業", "3008": "光電業", "2327": "電子零組件業", "2383": "電子零組件業", "3037": "電子零組件業",
            "2317": "其他電子業", "2354": "其他電子業",
            "2881": "金融保險業", "2882": "金融保險業", "2891": "金融保險業",
            "2603": "航運業", "2609": "航運業", "2615": "航運業",
        }

        for code, info in codes.items():
            if info.type in ['股票', 'ETF']:
                suffix = '.TW' if info.market == '上市' else '.TWO'
                full = f"{code}{suffix}"
                stock_map[full] = f"{full} {info.name}"
                
                if code in custom_sector_override: group = custom_sector_override[code]
                else: group = info.group if info.group else info.type
                if not group: group = "其他"
                
                if group not in industry_map: industry_map[group] = []
                industry_map[group].append(full)
                code_to_industry[code] = group
                
        return stock_map, industry_map, code_to_industry
    except: return {}, {}, {}

stock_map, industry_map, code_to_industry_map = get_tw_stock_list()

# 【第一層】：抓取台灣證券交易所/櫃買中心官方 API (獲取絕對精確的 PE, PB, Yield)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_twse_official_ratios():
    twse_data = {}
    try:
        res = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap14_L", timeout=10, verify=False)
        if res.status_code == 200:
            for item in res.json():
                code = str(item.get('證券代號', '')).strip()
                pe = pd.to_numeric(item.get('本益比', ''), errors='coerce')
                pb = pd.to_numeric(item.get('股價淨值比', ''), errors='coerce')
                yld = pd.to_numeric(item.get('殖利率(%)', ''), errors='coerce') / 100.0 if item.get('殖利率(%)') else np.nan
                twse_data[code] = {'pe': pe if pe > 0 else np.nan, 'pb': pb, 'yield': yld}
    except: pass
    try:
        res2 = requests.get("https://www.tpex.org.tw/openapi/v1/t187ap14_O", timeout=10, verify=False)
        if res2.status_code == 200:
            for item in res2.json():
                code = str(item.get('證券代號', '')).strip()
                pe = pd.to_numeric(item.get('本益比', ''), errors='coerce')
                pb = pd.to_numeric(item.get('股價淨值比', ''), errors='coerce')
                yld = pd.to_numeric(item.get('殖利率(%)', ''), errors='coerce') / 100.0 if item.get('殖利率(%)') else np.nan
                twse_data[code] = {'pe': pe if pe > 0 else np.nan, 'pb': pb, 'yield': yld}
    except: pass
    return twse_data

# 【第二層與第三層】：Yahoo Raw API & WantGoo API 聚合器
def get_robust_fundamentals(code, session, twse_data):
    fund = {'pe': np.nan, 'pb': np.nan, 'yield': np.nan, 'roe': np.nan, 'rev_growth': np.nan,
            'eps_growth': np.nan, 'gross_margins': np.nan, 'fcf': np.nan, 'market_cap': np.nan,
            'debt_to_equity': np.nan, 'beta': np.nan}
            
    # 1. 官方資料庫優先 (確保估值指標不漏接)
    if code in twse_data:
        fund['pe'] = twse_data[code].get('pe', np.nan)
        fund['pb'] = twse_data[code].get('pb', np.nan)
        fund['yield'] = twse_data[code].get('yield', np.nan)

    # 2. Yahoo Finance 底層原生 JSON API (跳過 yfinance 庫的 Cookie 限制)
    try:
        yahoo_code = f"{code}.TW" if len(code) == 4 else code
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{yahoo_code}?modules=defaultKeyStatistics,financialData,summaryDetail"
        res = session.get(url, timeout=4)
        if res.status_code == 200:
            data = res.json().get('quoteSummary', {}).get('result', [])[0]
            if 'financialData' in data:
                fd = data['financialData']
                fund['roe'] = fd.get('returnOnEquity', {}).get('raw', np.nan)
                fund['gross_margins'] = fd.get('grossMargins', {}).get('raw', np.nan)
                fund['rev_growth'] = fd.get('revenueGrowth', {}).get('raw', np.nan)
                fund['fcf'] = fd.get('freeCashflow', {}).get('raw', np.nan)
                fund['debt_to_equity'] = fd.get('debtToEquity', {}).get('raw', np.nan)
            if 'defaultKeyStatistics' in data:
                dks = data['defaultKeyStatistics']
                fund['eps_growth'] = dks.get('earningsQuarterlyGrowth', {}).get('raw', np.nan)
                fund['beta'] = dks.get('beta', {}).get('raw', np.nan)
                if pd.isna(fund['pb']): fund['pb'] = dks.get('priceToBook', {}).get('raw', np.nan)
            if 'summaryDetail' in data:
                sd = data['summaryDetail']
                fund['market_cap'] = sd.get('marketCap', {}).get('raw', np.nan)
                if pd.isna(fund['pe']): fund['pe'] = sd.get('trailingPE', {}).get('raw', np.nan)
                if pd.isna(fund['yield']): fund['yield'] = sd.get('dividendYield', {}).get('raw', np.nan)
    except: pass

    # 3. 玩股網 (WantGoo) 終極備援 (若 Yahoo 當機，從此處救援 ROE 與 毛利)
    if pd.isna(fund['roe']) or pd.isna(fund['gross_margins']):
        try:
            url_wg = f"https://www.wantgoo.com/investrue/api/v1/stock/{code}/financial-ratios"
            res_wg = session.get(url_wg, headers={'X-Requested-With': 'XMLHttpRequest'}, timeout=3)
            if res_wg.status_code == 200:
                wg_data = res_wg.json()
                if isinstance(wg_data, list) and len(wg_data) > 0:
                    latest = wg_data[0]
                    if pd.isna(fund['roe']): fund['roe'] = latest.get('returnOnEquity', np.nan) / 100.0
                    if pd.isna(fund['gross_margins']): fund['gross_margins'] = latest.get('grossMargin', np.nan) / 100.0
        except: pass

    return fund

def sanitize_data(df):
    if df.empty: return df
    if 'yield' in df.columns: df['yield'] = df['yield'].apply(lambda x: x/100 if x > 20 else x)
    return df

def calculate_implied_growth(price, eps, r=0.10, terminal_g=0.02, years=10):
    if pd.isna(price) or pd.isna(eps) or eps <= 0 or price <= 0: return np.nan
    low, high = -0.99, 3.0 
    tolerance = 0.01 
    for _ in range(100): 
        mid = (low + high) / 2
        pv = sum([eps * ((1 + mid) ** t) / ((1 + r) ** t) for t in range(1, years + 1)])
        tv = (eps * ((1 + mid) ** years) * (1 + terminal_g)) / (r - terminal_g)
        calc_price = pv + tv / ((1 + r) ** years)
        diff = calc_price - price
        if abs(diff) < tolerance: return mid
        if diff > 0: high = mid
        else: low = mid
    return (low + high) / 2

# --- 7. 批量掃描 (核心聚合器) ---
@st.cache_data(ttl=300, show_spinner=False) # 5分鐘快取，確保拿到 2026 最新資料
def batch_scan_stocks(stock_list):
    results = []
    history_map = {} 
    RISK_FREE_RATE = 0.015 
    
    global_session = create_resilient_session()
    twse_official_data = fetch_twse_official_ratios()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 使用多執行緒擷取
        future_to_stock = {}
        for s in stock_list:
            code_base = s.split(' ')[0]
            future_to_stock[executor.submit(get_robust_fundamentals, code_base.split('.')[0], global_session, twse_official_data)] = s
        
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            code = stock_str.split(' ')[0].split('.')[0]
            name = code 
            if len(stock_str.split(' ')) > 1: name = stock_str.split(' ')[1]

            fund_data = future.result()
            
            # 使用 yfinance 僅抓取歷史 K 線來算波動率、夏普、MDD
            price = np.nan; volatility = np.nan; sharpe = np.nan; mdd = np.nan; ma_bias = np.nan
            try:
                # 不餵入客製化 session 給 yfinance 歷史，避免破壞其內部防護
                ticker = yf.Ticker(stock_str.split(' ')[0])
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    history_map[code] = hist 
                    closes = hist['Close']
                    if len(closes) > 10:
                        price = float(closes.iloc[-1])
                        volatility = closes.pct_change().std() * (252**0.5)
                        ma60 = closes.rolling(60).mean().iloc[-1]
                        if not pd.isna(ma60): ma_bias = (price / ma60) - 1
                        
                        ret_6m = (closes.iloc[-1] / closes.iloc[0]) - 1
                        ann_ret = ret_6m * 2 
                        if volatility > 0 and not pd.isna(volatility):
                            sharpe = (ann_ret - RISK_FREE_RATE) / volatility
                            
                        roll_max = closes.cummax()
                        drawdown = closes / roll_max - 1.0
                        mdd = drawdown.min()
            except: pass

            # 聚合運算：PEG 與 FCF Yield
            fcf_yield = np.nan
            if not pd.isna(fund_data['fcf']) and not pd.isna(fund_data['market_cap']) and fund_data['market_cap'] > 0:
                fcf_yield = (fund_data['fcf'] / fund_data['market_cap']) * 100
                
            peg = np.nan
            if not pd.isna(fund_data['pe']) and not pd.isna(fund_data['eps_growth']) and fund_data['eps_growth'] > 0:
                peg = fund_data['pe'] / (fund_data['eps_growth'] * 100)
                
            # 估算 EPS 用於 DCF
            t_eps = np.nan
            if not pd.isna(price) and not pd.isna(fund_data['pe']) and fund_data['pe'] > 0:
                t_eps = price / fund_data['pe']
            implied_g = calculate_implied_growth(price, t_eps)

            industry = code_to_industry_map.get(code, '未分類')
            if code.startswith('00'): industry = 'ETF'

            if not pd.isna(price):
                results.append({
                    '代號': code, '名稱': name, 'close_price': price,
                    'pe': fund_data['pe'], 'pb': fund_data['pb'], 
                    'yield': fund_data['yield'] * 100 if not pd.isna(fund_data['yield']) else np.nan, 
                    'roe': fund_data['roe'], 
                    'rev_growth': fund_data['rev_growth'] * 100 if not pd.isna(fund_data['rev_growth']) else np.nan, 
                    'eps_growth': fund_data['eps_growth'] * 100 if not pd.isna(fund_data['eps_growth']) else np.nan, 
                    'gross_margins': fund_data['gross_margins'] * 100 if not pd.isna(fund_data['gross_margins']) else np.nan,
                    'fcf_yield': fcf_yield, 'de_ratio': fund_data['debt_to_equity'], 'beta': fund_data['beta'],
                    'sharpe': sharpe, 'mdd': mdd, 'implied_growth': implied_g,
                    'peg': peg, 'chips': 0, 'volatility': volatility, 'priceToMA60': ma_bias,
                    'industry': industry, 'full_symbol': stock_str
                })
    
    df = pd.DataFrame(results)
    cols = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'roe', 'rev_growth', 'eps_growth', 'gross_margins', 'fcf_yield', 'de_ratio', 'beta', 'sharpe', 'mdd', 'implied_growth', 'peg', 'chips', 'volatility', 'priceToMA60', 'industry']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df, history_map

# --- 8. 評分邏輯 (三大水桶：成長/防禦/中性) ---
def calculate_score(df, logic_type="Quant"):
    if df.empty: return df, None
    required_cols = ['pe', 'pb', 'yield', 'rev_growth', 'eps_growth', 'gross_margins', 'fcf_yield', 'de_ratio', 'beta', 'sharpe', 'mdd', 'implied_growth', 'peg', 'volatility', 'roe', 'priceToMA60']
    for col in required_cols:
        if col not in df.columns: df[col] = np.nan
    
    df_norm = df.copy()
    scores = []
    plans = []
    quality_tags = []
    fill_map = {c: 0 for c in required_cols}
    fill_map['pe'] = 50; fill_map['volatility'] = 0.5; fill_map['beta'] = 1.0; fill_map['de_ratio'] = 100
    calc_df = df.fillna(fill_map)

    for idx, row in calc_df.iterrows():
        total_score = 0
        total_weight = 0
        if logic_type == "Quant":
            config = {
                'Sharpe': {'col': 'sharpe', 'dir': 'max', 'w': 4.0, 'cat': '動能'},
                'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 2.0, 'cat': '風險'},
                'Beta': {'col': 'beta', 'dir': 'mid', 'w': 1.0, 'cat': '風險'}, 
            }
        else:
            config = {
                'ROE': {'col': 'roe', 'dir': 'max', 'w': 2.0, 'cat': '財報'},
                'GrossMargin': {'col': 'gross_margins', 'dir': 'max', 'w': 1.5, 'cat': '財報'},
                'FCF_Yield': {'col': 'fcf_yield', 'dir': 'max', 'w': 2.0, 'cat': '價值'},
                'DE_Ratio': {'col': 'de_ratio', 'dir': 'min', 'w': 1.5, 'cat': '風險'},
                'EPS_Growth': {'col': 'eps_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'},
            }
            
        for name, setting in config.items():
            val = row.get(setting['col'])
            all_vals = calc_df[setting['col']]
            rank = all_vals.rank(pct=True).get(idx, 0.5)
            if setting['dir'] == 'max': norm = rank
            elif setting['dir'] == 'min': norm = 1 - rank
            else: norm = 1 - abs(rank - 0.5)*2 
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm * 100
            total_score += norm * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final, 1))
        
        q_tag = ""
        if logic_type == "Quant":
            sh = row.get('sharpe', 0)
            vol = row.get('volatility', 1)
            b = row.get('beta', 1.0)
            mdd = row.get('mdd', 0)
            if pd.isna(b): b = 1.0
            
            if mdd < -0.25: 
                q_tag = "回撤過大 (剔除)"
                plans.append("跌破 25% 防線")
                scores[-1] = 0 
            elif sh > 1.0 and b >= 1.0: 
                q_tag = "成長型資產"
                plans.append("配置主動能")
            elif sh > 1.0 and b < 1.0: 
                q_tag = "防禦型資產"
                plans.append("配置穩定基石")
            elif sh > 0 and vol < 0.25: 
                q_tag = "中性資產"
                plans.append("降低組合波動")
            elif sh < 0: 
                q_tag = "風險報酬不對等"
                plans.append("避開標的")
            else: 
                q_tag = "中立觀望"
                plans.append("中立觀望")
            
        else: 
            roe = row.get('roe', 0)
            gm = row.get('gross_margins', 0)
            de = row.get('de_ratio', 100)
            fcf = row.get('fcf_yield', 0)
            mdd = row.get('mdd', 0)
            
            if mdd < -0.25:
                q_tag = "回撤過大 (剔除)"
                plans.append("跌破 25% 防線")
                scores[-1] = 0
            elif roe > 15 and gm > 40 and de < 100 and fcf > 0: 
                q_tag = "護城河優良"
                plans.append("價值浮現")
            elif de > 200 or fcf < -5: 
                q_tag = "財務風險"
                plans.append("避開標的")
            else: 
                q_tag = "中立觀望"
                if final > 60: plans.append("合理估值")
                else: plans.append("中立觀望")
            
        quality_tags.append(q_tag)
            
    df['Score'] = scores
    df['Strategy'] = plans
    df['Quality'] = quality_tags
    return df.sort_values('Score', ascending=False), df_norm

# --- 9. 繪圖函數 ---
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
    for k, v in cats.items():
        if counts[k] > 0: radar[k] = v / counts[k]
        else: radar[k] = 50
    return radar

def plot_radar_chart_ui(title, radar_data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(radar_data.values()), theta=list(radar_data.keys()),
        fill='toself', name=title, line_color='#3B82F6', fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=30, r=30), height=250, font=dict(color='#9CA3AF')
    )
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

    fig.update_layout(
        title=dict(text=f"<b>中長期趨勢研判</b><br><span style='font-size:12px; color:#9CA3AF'>季線乖離 {bias_pct:.1f}% | {status_text}</span>", font=dict(color='#F8FAFC', size=14), y=0.95),
        xaxis=dict(showgrid=False, linecolor='#334155', tickfont=dict(color='#64748B')),
        yaxis=dict(showgrid=True, gridcolor='#1E293B', tickfont=dict(color='#64748B')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=20, l=0, r=0), height=250,
        showlegend=False, hovermode="x unified"
    )
    return fig

# --- 10. AI 與 PDF ---
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
        session = create_resilient_session()
        r = session.post(url, headers=headers, json=data, timeout=60)
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
        ['夏普值 (Sharpe)', safe_str(stock_data.get('sharpe')), '波動率 (Volatility)', safe_str(stock_data.get('volatility')*100 if not pd.isna(stock_data.get('volatility')) else np.nan, "{:.1f}%")],
        ['Beta (風險係數)', safe_str(stock_data.get('beta'), "{:.2f}"), '最大回撤 (MDD)', safe_str(stock_data.get('mdd')*100 if not pd.isna(stock_data.get('mdd')) else np.nan, "{:.1f}%")],
        ['ROE (權益報酬)', safe_str(stock_data.get('roe')*100 if not pd.isna(stock_data.get('roe')) else np.nan, "{:.1f}%"), '毛利率 (Gross Margin)', safe_str(stock_data.get('gross_margins'), "{:.1f}%")],
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
    except Exception as e: print(e)
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

def get_tag_explanation(row, logic_type):
    tag = row.get('Quality', '')
    sharpe = row.get('sharpe', 0)
    roe = (row.get('roe', 0)*100) if not pd.isna(row.get('roe')) else 0
    gm = row.get('gross_margins', 0)
    fcf = row.get('fcf_yield', 0)
    de = row.get('de_ratio', 0)
    mdd = row.get('mdd', 0)
    
    if tag == "回撤過大 (剔除)": return f"**為什麼被系統剔除？**<br>此標的過去半年內的最大回撤 (MDD) 高達 **{mdd*100:.1f}%**。依據嚴格的量化風控紀律，任何跌破 25% 容忍防線的資產，無論其基本面或夏普值多麼亮眼，都可能造成投資組合的永久性資本損傷，因此被系統強制剔除。"
    
    if logic_type == "Quant":
        if tag == "成長型資產": return f"**為什麼被系統評定為「🚀 成長型資產」？**<br>此標的近期年化夏普值高達 **{sharpe:.2f}** 且 Beta 大於 1。依據林哲群教授《紀律長贏》觀念，其報酬率累積速度遠大於風險的增加，屬於位於效率前緣的高回報攻擊動能。"
        elif tag == "防禦型資產": return f"**為什麼被系統評定為「🛡️ 防禦型資產」？**<br>此標的近期年化夏普值高達 **{sharpe:.2f}**，且 Beta 值小於 1。具備低於大盤的波動特性與高性價比報酬，是優化投資組合下檔風險的防禦基石。"
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
    if not selected_codes:
        first_code = pool_df.loc[pool_df[metric_col].idxmax(), '代號']
        selected_codes.append(first_code)
        candidates.remove(first_code)
        
    while len(selected_codes) < (len(initial_selected or []) + target_n) and candidates:
        best_score = -999999
        best_code = None
        for code in candidates:
            base_val = pool_df[pool_df['代號'] == code][metric_col].values[0]
            norm_metric = (base_val - min_metric) / (max_metric - min_metric)
            corrs = []
            for sel_code in selected_codes:
                if code in returns_df.columns and sel_code in returns_df.columns:
                    c = returns_df[code].corr(returns_df[sel_code])
                    if not pd.isna(c): corrs.append(c)
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
                
            vol_penalty = 0
            if port_vol > target_vol_max:
                vol_penalty = (port_vol - target_vol_max) * 100.0 
            obj_score = norm_metric - corr_penalty - vol_penalty 
            if obj_score > best_score:
                best_score = obj_score
                best_code = code
        if best_code:
            selected_codes.append(best_code)
            candidates.remove(best_code)
        else: break
    return selected_codes

# --- 14. 自選組合自動監控訊號 ---
def generate_custom_signal(row, regime):
    s = row.get('sharpe', 0)
    mdd = row.get('mdd', 0)
    b = row.get('beta', 1.0)
    tag = row.get('Quality', '')
    
    if mdd < -0.25: return "🛑 強制停損 (MDD破防)"
    if s < 0: return "⚠️ 劣勢資產 (建議剔除)"
    
    if regime == "Bull":
        if tag == "成長型資產" and s > 2.0 and mdd > -0.10: return "🔥 強勢加碼 (放大獲利)"
        if tag == "防禦型資產" or tag == "中性資產": return "📉 逢高減碼 (釋放資金)"
        return "⚖️ 持有續抱"
    else: 
        if b > 1.2: return "📉 降持高波 (收回現金)"
        if tag == "防禦型資產" or tag == "中性資產": return "🛡️ 逢低加碼 (鞏固底盤)"
        return "⚖️ 持有觀望"

# --- 15. 主程式介面 ---
with st.sidebar:
    st.title("⚙️ 終端控制面板")
    st.subheader("分析核心模型切換")
    logic_choice_tw = st.radio(
        "選擇運算引擎", 
        ["📊 量化風控模型 (夏普值/波動率/Beta)", "👴 價值護城河 (高ROE/毛利/現金流)"],
        index=0, label_visibility="collapsed"
    )
    st.session_state['current_logic'] = "Quant" if "量化" in logic_choice_tw else "Buffett"
    st.markdown("---")
    
    scan_mode = st.radio("篩選維度", ["市場焦點策略", "產業族群板塊", "台灣 ETF 專區", "自訂代碼輸入"])
    target_stocks = []
    
    if st.session_state['current_logic'] == "Buffett":
        strategies = {
            "台灣50 (護城河權值)": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2412.TW", "1301.TW"],
            "高股息與穩健 (現金流)": ["2454.TW", "2303.TW", "2357.TW", "1101.TW", "2891.TW"],
            "金融保險 (防禦收息)": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "5880.TW", "2884.TW"],
            "傳產與航運 (循環價值)": ["2603.TW", "2609.TW", "2002.TW", "1301.TW", "1303.TW", "1605.TW"]
        }
    else: 
        strategies = {
            "動能爆發 (高夏普)": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "3661.TW"],
            "低波動防禦 (低標準差)": ["2412.TW", "3045.TW", "2881.TW", "2892.TW", "1101.TW"],
            "蘋果與消費電 (Beta循環)": ["2330.TW", "2317.TW", "3008.TW", "4938.TW", "2313.TW"],
            "電動車與車電 (趨勢Beta)": ["2308.TW", "2317.TW", "6235.TW", "1536.TW", "5425.TW"]
        }
        
    etf_strategies = {
        "市值型 ETF (大盤連動)": ["0050.TW", "006208.TW", "00692.TW", "00881.TW"],
        "高股息 ETF (穩定配息)": ["0056.TW", "00878.TW", "00919.TW", "00929.TW", "00713.TW"],
        "科技與半導體主題": ["00891.TW", "00892.TW", "00881.TW", "00830.TW"],
        "海外與美股連結": ["00757.TW", "00646.TW", "00830.TW", "00662.TW"]
    }
    
    if scan_mode == "自訂代碼輸入":
        default = ["2330.TW 台積電", "2454.TW 聯發科", "0050.TW 元大台灣50"]
        options = sorted(list(stock_map.values())) if stock_map else default
        selected = st.multiselect("搜尋上市櫃標的", options, default=[s for s in default if s in options])
        manual = st.text_input("快速輸入代號 (如 2317)", "")
        target_stocks = selected
        if manual: target_stocks.append(f"{manual}.TW")
    elif scan_mode == "產業族群板塊":
        ind_keys = sorted(list(industry_map.keys()))
        selected_inds = st.multiselect("板塊選擇 (支援複選)", ["[ALL] 載入全部板塊"] + ind_keys)
        if "[ALL] 載入全部板塊" in selected_inds:
            for k in ind_keys: target_stocks.extend(industry_map[k])
        else:
            for k in selected_inds: target_stocks.extend(industry_map[k])
    elif scan_mode == "台灣 ETF 專區":
        etf_keys = list(etf_strategies.keys())
        selected_etfs = st.multiselect("ETF 分類 (支援複選)", ["[ALL] 載入全部 ETF"] + etf_keys, default=[etf_keys[0]])
        if "[ALL] 載入全部 ETF" in selected_etfs:
            for k in etf_keys: target_stocks.extend(etf_strategies[k])
        else:
            for k in selected_etfs: target_stocks.extend(etf_strategies[k])
    else: 
        strat_keys = list(strategies.keys())
        selected_strats = st.multiselect(f"推薦策略池 (支援複選)", ["[ALL] 載入全部策略"] + strat_keys, default=[strat_keys[0]])
        if "[ALL] 載入全部策略" in selected_strats:
            for k in strat_keys: target_stocks.extend(strategies[k])
        else:
            for k in selected_strats: target_stocks.extend(strategies[k])

    target_stocks = list(dict.fromkeys(target_stocks)) 

    if st.button("🚀 啟動終端運算", type="primary", use_container_width=True):
        st.session_state['scan_finished'] = False
        st.session_state['panel_page'] = 1 
        with st.spinner(f"正在聚合最新財報與量化數據 (共 {len(target_stocks)} 檔)..."):
            raw, hist_map = batch_scan_stocks(target_stocks)
            raw = sanitize_data(raw)
            st.session_state['raw_data'] = raw
            st.session_state['history_storage'] = hist_map 
            st.session_state['scan_finished'] = True
            st.rerun()

# --- 主畫面區 ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 台股量化與價值分析終端")
    logic_badge = "量化風控引擎" if st.session_state['current_logic'] == "Quant" else "價值護城河引擎"
    st.caption(f"STATUS: ONLINE | ENGINE: **{logic_badge}** | ALGO: 多源數據聚合與財報滿血版")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    current_logic = st.session_state['current_logic']
    
    if df.empty:
        st.error("檢索結果為空，請確認資料源連線狀態或代碼有效性。")
    else:
        final_df, df_norm = calculate_score(df, logic_type=current_logic)
        
        # =========================================================
        # 🔍 終端快速篩選器 (Quick Filter)
        # =========================================================
        st.subheader("🏆 終端檢索清單")
        
        unique_tags = final_df['Quality'].dropna().unique().tolist()
        def tag_sort_key(t):
            if "成長" in t or "護城河" in t: return 0
            if "防禦" in t: return 1
            if "中性" in t: return 2
            if "觀望" in t: return 3
            return 4
        unique_tags = sorted(unique_tags, key=tag_sort_key)
        
        filter_options = ["[ALL] 顯示全部標的"] + unique_tags
        selected_filter = st.radio("🎯 快速分類篩選：", filter_options, horizontal=True)
        
        if selected_filter != "[ALL] 顯示全部標的":
            filtered_df = final_df[final_df['Quality'] == selected_filter]
        else:
            filtered_df = final_df
            
        st.caption("💡 點選下方資料列即可查看「深度解析」並【加入自選組合】。")
        
        df_event = st.dataframe(
            filtered_df[['代號', '名稱', 'industry', 'Score', 'Quality', 'Strategy', 'sharpe', 'mdd', 'roe', 'implied_growth']],
            column_config={
                "industry": st.column_config.TextColumn("產業/板塊"),
                "Score": st.column_config.ProgressColumn("綜合評分", min_value=0, max_value=100, format="%.1f"),
                "Quality": st.column_config.TextColumn("系統標籤"),
                "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"),
                "mdd": st.column_config.NumberColumn("最大回撤", format="%.1f%%"),
                "roe": st.column_config.NumberColumn("ROE", format="%.1f"),
                "implied_growth": st.column_config.NumberColumn("隱含成長", format="%.2f"),
            },
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row" 
        )
        st.markdown("---")
        
        selected_rows = df_event.selection.rows
        
        if selected_rows:
            sel_idx = selected_rows[0]
            sel_row = filtered_df.iloc[sel_idx]
            
            c_title, c_add = st.columns([3, 1])
            with c_title:
                st.subheader(f"💡 系統標籤深度解析：{sel_row['名稱']} ({sel_row['代號']})")
            with c_add:
                sel_code = sel_row['代號']
                is_in_port = sel_code in st.session_state['my_portfolio']
                btn_txt = "❌ 從自選移除" if is_in_port else "➕ 加入自選戰略組合"
                if st.button(btn_txt, use_container_width=True):
                    if is_in_port: st.session_state['my_portfolio'].remove(sel_code)
                    else: st.session_state['my_portfolio'].append(sel_code)
                    st.rerun()

            explanation = get_tag_explanation(sel_row, current_logic)
            st.markdown(f"<div class='explain-box'>{explanation}</div>", unsafe_allow_html=True)
            st.subheader("🎯 個股深度檢驗面板 (單檔檢視)")
            st.info("📌 目前為「單檔檢視模式」。若要恢復分頁瀏覽模式，請在上方清單中再次點擊以取消選取。")
            display_df = filtered_df.iloc[[sel_idx]]
        else:
            st.subheader("🎯 個股深度檢驗面板 (全覽分頁)")
            total_stocks = len(filtered_df)
            items_per_page = 5
            total_pages = max(1, (total_stocks - 1) // items_per_page + 1) if total_stocks > 0 else 1
            
            if st.session_state['panel_page'] > total_pages:
                st.session_state['panel_page'] = total_pages
            
            start_idx = (st.session_state['panel_page'] - 1) * items_per_page
            end_idx = start_idx + items_per_page
            display_df = filtered_df.iloc[start_idx:end_idx]
            
        def safe_num(val): return 0 if (pd.isna(val) or val is None) else val
        def get_color_class(val, inverse=False):
            if pd.isna(val) or val == 0: return ""
            if inverse: return "q-up" if val < 0 else "q-down"
            return "q-up" if val > 0 else "q-down"

        for idx, row in display_df.iterrows():
            code = row['代號']
            with st.container():
                industry_tag = f"<span class='tag tag-sector'>{row['industry']}</span>"
                q_tag = row['Quality']
                if q_tag in ["成長型資產", "防禦型資產", "護城河優良", "Quality"]: quality_tag = f"<span class='tag tag-moat'>💎 {q_tag}</span>"
                elif q_tag in ["中性資產"]: quality_tag = f"<span class='tag tag-logic'>⚖️ {q_tag}</span>"
                elif q_tag in ["風險報酬不對等", "財務風險", "回撤過大 (剔除)"]: quality_tag = f"<span class='tag tag-risk'>⚠️ {q_tag}</span>"
                else: quality_tag = ""
                logic_tag = f"<span class='tag tag-logic'>{logic_badge}</span>"
                
                st.markdown(f"""<div class='stock-card'>
<div class='card-header'>
<div><span class='header-title'>{row['名稱']} ({code})</span><span class='header-price'>${row['close_price']}</span></div>
<div>{logic_tag}{industry_tag}{quality_tag}</div>
</div>""", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 1.8, 1.5])
                with c1:
                    orig_idx = df_norm.index[df_norm['代號'] == code]
                    if len(orig_idx) > 0:
                        st.plotly_chart(plot_radar_chart_ui(row['名稱'], get_radar_data(df_norm.loc[orig_idx[0]])), use_container_width=True)
                with c2:
                    sharpe_val = safe_num(row.get('sharpe'))
                    sh_color = "q-up" if sharpe_val > 1 else ("q-down" if sharpe_val < 0 else "")
                    rev_val = safe_num(row.get('rev_growth'))
                    eps_val = safe_num(row.get('eps_growth'))
                    mdd_val = safe_num(row.get('mdd'))
                    mdd_color = "q-up" if mdd_val > -0.15 else ("q-down" if mdd_val < -0.25 else "q-neu")
                    
                    st.markdown(f"""<div class='quote-board'>
<div class='quote-item'><span class='q-label'>夏普值 (Sharpe)</span><span class='q-val {sh_color}'>{sharpe_val:.2f}</span></div>
<div class='quote-item'><span class='q-label'>波動率 (Volatility)</span><span class='q-val'>{safe_num(row.get('volatility'))*100:.1f}%</span></div>
<div class='quote-item'><span class='q-label'>Beta (風險係數)</span><span class='q-val'>{safe_num(row.get('beta')):.2f}</span></div>
<div class='quote-item'><span class='q-label'>最大回撤 (MDD)</span><span class='q-val {mdd_color}'>{mdd_val*100:.1f}%</span></div>
<div class='quote-item'><span class='q-label'>ROE (權益報酬)</span><span class='q-val'>{"N/A" if pd.isna(row.get('roe')) else f"{safe_num(row.get('roe'))*100 if row.get('roe') and row.get('roe')<1 else safe_num(row.get('roe')):.1f}%"}</span></div>
<div class='quote-item'><span class='q-label'>毛利率 (Gross Mgn)</span><span class='q-val'>{"N/A" if pd.isna(row.get('gross_margins')) else f"{safe_num(row.get('gross_margins')):.1f}%"}</span></div>
<div class='quote-item'><span class='q-label'>營收 YoY</span><span class='q-val {get_color_class(rev_val)}'>{rev_val:.1f}%</span></div>
<div class='quote-item'><span class='q-label'>EPS YoY</span><span class='q-val {get_color_class(eps_val)}'>{eps_val:.1f}%</span></div>
<div class='quote-item'><span class='q-label'>殖利率 (Yield)</span><span class='q-val'>{"N/A" if pd.isna(row.get('yield')) else f"{safe_num(row.get('yield')):.2f}%"}</span></div>
<div class='quote-item'><span class='q-label'>本益比 (P/E)</span><span class='q-val'>{"N/A" if pd.isna(row.get('pe')) else f"{safe_num(row.get('pe')):.2f}"}</span></div>
<div class='quote-item'><span class='q-label'>PEG 估值</span><span class='q-val'>{"N/A" if pd.isna(row.get('peg')) else f"{safe_num(row.get('peg')):.2f}"}</span></div>
<div class='quote-item'><span class='q-label'>FCF 收益率</span><span class='q-val'>{"N/A" if pd.isna(row.get('fcf_yield')) else f"{safe_num(row.get('fcf_yield')):.2f}%"}</span></div>
</div>""", unsafe_allow_html=True)
                    
                    implied_g = row.get('implied_growth', np.nan)
                    actual_g = row.get('eps_growth', 0) / 100 
                    if not pd.isna(implied_g):
                        if implied_g > actual_g + 0.1: dcf_status = "<span class='q-down'>🔴 預估過熱</span>"
                        elif implied_g < actual_g - 0.05: dcf_status = "<span class='q-up'>🟢 具安全邊際</span>"
                        else: dcf_status = "<span class='q-neu'>🟡 估值公允</span>"
                        st.markdown(f"""<div class='dcf-panel'>
<div style='font-size: 0.8rem; color: #9CA3AF; font-weight: 600; text-transform: uppercase;'>REVERSE DCF 估值模型 (r=10%, TV=2%)</div>
<div style='display: flex; justify-content: space-between; margin-top: 6px; align-items: baseline;'>
<span style='color: #F8FAFC;'>市場隱含期望: <b style='font-size: 1.2rem; font-family: Consolas;'>{implied_g*100:.1f}%</b></span>
<span style='font-weight: 700; font-size: 0.95rem;'>{dcf_status}</span>
</div></div>""", unsafe_allow_html=True)
                    
                    c_btn1, c_btn2 = st.columns(2)
                    if code not in st.session_state['ai_results']:
                        with c_btn1:
                            if st.button(f"🧠 執行 AI 洞察", key=f"ai_{code}"):
                                current_today = datetime.now().strftime('%Y年%m月%d日')
                                if current_logic == "Quant":
                                    l_name = "量化風控模型"
                                    l_desc = "請以林哲群教授理論評估，注意「最大回撤(MDD)」是否破25%。解釋其在「成長/防禦/中性」三類資產的戰略定位。"
                                else:
                                    l_name = "價值護城河模型"
                                    l_desc = "優先考量ROE、毛利率、現金流，並以最大回撤防禦極端風險。"
                                p_txt = AI_PROMPT_TEMPLATE.replace("[LOGIC_NAME]", l_name) \
                                    .replace("[LOGIC_DESC]", l_desc) \
                                    .replace("[STOCK]", row['名稱']) \
                                    .replace("[SECTOR]", str(row['industry'])) \
                                    .replace("[CURRENT_DATE]", current_today) \
                                    .replace("[SHARPE]", str(safe_num(row.get('sharpe')))) \
                                    .replace("[VOL]", str(safe_num(row.get('volatility'))*100)) \
                                    .replace("[BETA]", str(safe_num(row.get('beta')))) \
                                    .replace("[MDD]", str(safe_num(row.get('mdd'))*100)) \
                                    .replace("[ROE]", str(safe_num(row.get('roe'))*100 if row.get('roe') and row.get('roe')<1 else safe_num(row.get('roe')))) \
                                    .replace("[GM]", str(safe_num(row.get('gross_margins')))) \
                                    .replace("[FCF_Y]", str(safe_num(row.get('fcf_yield')))) \
                                    .replace("[DE]", str(safe_num(row.get('de_ratio')))) \
                                    .replace("[EPS_G]", str(safe_num(row.get('eps_growth')))) \
                                    .replace("[PE]", str(safe_num(row.get('pe')))) \
                                    .replace("[IMPLIED_G]", str(safe_num(implied_g)*100)) \
                                    .replace("[MA_BIAS]", str(safe_num(row.get('priceToMA60'))*100))
                                an = call_ai(p_txt)
                                st.session_state['ai_results'][code] = an
                                st.rerun() 
                    else:
                        with c_btn1:
                            if st.button(f"🔄 更新 AI 洞察", key=f"ai_re_{code}"):
                                current_today = datetime.now().strftime('%Y年%m月%d日')
                                if current_logic == "Quant":
                                    l_name = "量化風控模型"
                                    l_desc = "請以林哲群教授理論評估，注意「最大回撤(MDD)」是否破25%。解釋其在「成長/防禦/中性」三類資產的戰略定位。"
                                else:
                                    l_name = "價值護城河模型"
                                    l_desc = "優先考量ROE、毛利率、現金流，並以最大回撤防禦極端風險。"
                                p_txt = AI_PROMPT_TEMPLATE.replace("[LOGIC_NAME]", l_name) \
                                    .replace("[LOGIC_DESC]", l_desc) \
                                    .replace("[STOCK]", row['名稱']) \
                                    .replace("[SECTOR]", str(row['industry'])) \
                                    .replace("[CURRENT_DATE]", current_today) \
                                    .replace("[SHARPE]", str(safe_num(row.get('sharpe')))) \
                                    .replace("[VOL]", str(safe_num(row.get('volatility'))*100)) \
                                    .replace("[BETA]", str(safe_num(row.get('beta')))) \
                                    .replace("[MDD]", str(safe_num(row.get('mdd'))*100)) \
                                    .replace("[ROE]", str(safe_num(row.get('roe'))*100 if row.get('roe') and row.get('roe')<1 else safe_num(row.get('roe')))) \
                                    .replace("[GM]", str(safe_num(row.get('gross_margins')))) \
                                    .replace("[FCF_Y]", str(safe_num(row.get('fcf_yield')))) \
                                    .replace("[DE]", str(safe_num(row.get('de_ratio')))) \
                                    .replace("[EPS_G]", str(safe_num(row.get('eps_growth')))) \
                                    .replace("[PE]", str(safe_num(row.get('pe')))) \
                                    .replace("[IMPLIED_G]", str(safe_num(implied_g)*100)) \
                                    .replace("[MA_BIAS]", str(safe_num(row.get('priceToMA60'))*100))
                                an = call_ai(p_txt)
                                st.session_state['ai_results'][code] = an
                                st.rerun()
                        with c_btn2:
                            pdf_payload = row.to_dict()
                            pdf_payload['ai_analysis'] = st.session_state['ai_results'][code]
                            pdf = create_pdf(pdf_payload)
                            file_name_dl = f"{code} {row['名稱']} ({(row.get('full_symbol', code))})_Report.pdf"
                            st.download_button("📥 輸出 PDF 報告", pdf, file_name_dl, key=f"dl_{code}")

                with c3:
                    if code in hist_storage and not hist_storage[code].empty:
                        st.plotly_chart(plot_trend_dashboard(row['名稱'], hist_storage[code], row.get('priceToMA60', 0)), use_container_width=True)
                    else:
                        st.warning("無 K 線數據")

                if code in st.session_state['ai_results']:
                    st.markdown(f"<div class='ai-box'>{st.session_state['ai_results'][code]}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        if not selected_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
            with col_p1:
                if st.button("⬅️ 上一頁", use_container_width=True, disabled=(st.session_state['panel_page'] == 1)):
                    st.session_state['panel_page'] -= 1
                    st.rerun()
            with col_p2:
                st.markdown(f"<div style='text-align: center; color: #9CA3AF; padding-top: 8px;'>第 <b>{st.session_state['panel_page']}</b> 頁 / 共 <b>{total_pages}</b> 頁 (符合條件共 {total_stocks} 檔)</div>", unsafe_allow_html=True)
            with col_p3:
                if st.button("下一頁 ➡️", use_container_width=True, disabled=(st.session_state['panel_page'] == total_pages) or total_pages == 0):
                    st.session_state['panel_page'] += 1
                    st.rerun()

        # =========================================================
        # 🛠️ 戰略指揮中心：自選組合監控與多空情境推演
        # =========================================================
        st.markdown("---")
        st.subheader("🛠️ 戰略指揮中心：自選組合監控與多空情境推演")
        
        if not st.session_state['my_portfolio']:
            st.info("💡 目前您的自選戰略組合為空。請在上方「終端檢索清單」點選標的，並將其【➕ 加入自選戰略組合】以啟用監控與推演功能。")
        else:
            regime = st.radio("🌍 切換總經市場情境 (Market Regime Engine)", ["📈 多頭市場 (Bull Market) - 放大獲利", "📉 空頭/震盪市場 (Bear Market) - 著重防禦"], horizontal=True)
            
            my_port_codes = list(st.session_state['my_portfolio'])
            my_port_df = final_df[final_df['代號'].isin(my_port_codes)].copy()
            
            def assign_bucket(r):
                s = r.get('sharpe', 0)
                b = r.get('beta', 1.0)
                v = r.get('volatility', 1.0)
                mdd = r.get('mdd', 0)
                if pd.isna(b): b = 1.0
                if mdd < -0.25: return "🛑 剔除資產"
                if s > 1.0 and b >= 1.0: return "🚀 成長型資產"
                elif s > 1.0 and b < 1.0: return "🛡️ 防禦型資產"
                elif s > 0 and v < 0.25: return "⚖️ 中性資產"
                else: return "🟡 觀察資產"
                
            def get_signal(r):
                tag = r.get('Quality', '')
                return generate_custom_signal(r, "Bull" if "多頭" in regime else "Bear")
                
            my_port_df['資產屬性'] = my_port_df.apply(assign_bucket, axis=1)
            my_port_df['操作訊號'] = my_port_df.apply(get_signal, axis=1)
            
            st.markdown("<h5 style='color: #F8FAFC;'>即時監控與交易訊號表</h5>", unsafe_allow_html=True)
            st.dataframe(
                my_port_df[['代號', '名稱', '資產屬性', 'sharpe', 'beta', 'mdd', '操作訊號']].sort_values('sharpe', ascending=False),
                hide_index=True, use_container_width=True,
                column_config={
                    "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"),
                    "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                    "mdd": st.column_config.NumberColumn("最大回撤", format="%.1f%%"),
                }
            )
            
            count_grow = len(my_port_df[my_port_df['資產屬性'] == "🚀 成長型資產"])
            count_neu = len(my_port_df[my_port_df['資產屬性'] == "⚖️ 中性資產"])
            count_def = len(my_port_df[my_port_df['資產屬性'] == "🛡️ 防禦型資產"])
            count_bad = len(my_port_df[my_port_df['資產屬性'].isin(["🛑 剔除資產", "🟡 觀察資產"])])
            
            if "多頭" in regime:
                t_grow, t_neu, t_def, t_cash = 60, 25, 10, 5
                advice_text = "在多頭環境下，建議將**成長型資產配置提高至約 60%**，讓高 Sharpe 且具動能的標的持續放大獲利，同時避免過度分散。"
            else:
                t_grow, t_neu, t_def, t_cash = 30, 30, 25, 15
                advice_text = "在空頭或震盪環境下，需優先控制回撤。建議主動降低高 Beta 標的持倉，**將防禦型與中性資產合計拉高至 55%**，並保留充裕現金。"
            
            st.markdown(f"<div class='cmd-box'><b>📝 情境推演建議：</b><br>{advice_text}</div>", unsafe_allow_html=True)
            
            w_c1, w_c2, w_c3, w_c4 = st.columns(4)
            
            def render_bucket(col, title, current_count, target_pct):
                warning = ""
                if current_count == 0 and target_pct > 0:
                    warning = f"<br><span style='color:#EF4444; font-size:0.8rem;'>⚠️ 缺乏此類資產，建議自清單納入</span>"
                col.markdown(f"""
                <div style='background-color:#06090F; padding:15px; border-radius:4px; border:1px solid #1E293B;'>
                    <div style='color:#9CA3AF; font-size:0.9rem;'>{title}</div>
                    <div style='font-size:1.5rem; font-weight:bold; color:#F8FAFC;'>目標 {target_pct}%</div>
                    <div style='color:#3B82F6; font-size:0.85rem;'>目前自選檔數: {current_count} 檔</div>
                    {warning}
                </div>
                """, unsafe_allow_html=True)

            render_bucket(w_c1, "🚀 成長型資產", count_grow, t_grow)
            render_bucket(w_c2, "⚖️ 中性資產", count_neu, t_neu)
            render_bucket(w_c3, "🛡️ 防禦型資產", count_def, t_def)
            render_bucket(w_c4, "💵 現金部位", "-", t_cash)
            
            if count_bad > 0:
                st.warning(f"⚠️ 您的自選組合中有 {count_bad} 檔被判定為「劣勢/剔除」資產 (MDD 破防或夏普小於 0)，強烈建議依照上方訊號表進行減碼或停損！")

        # =========================================================
        # 💼 系統嚴選：自動化 10 檔模型專屬投資組合
        # =========================================================
        st.markdown("---")
        
        if current_logic == "Quant":
            st.subheader("💼 系統嚴選：【紀律長贏】效率前緣配置 (Automated MPT)")
            st.caption("依據林哲群教授理論：MDD < -25% 剔除。強制將 MPT 組合波動率控制於 **10% ~ 15%**。系統使用「4-3-3 均衡降維演算法」，自動為您抽出 4檔成長 + 3檔防禦 + 3檔中性 的終極抗震組合。")
            
            df_list = []
            for c, h_df in hist_storage.items():
                if not h_df.empty and 'Close' in h_df.columns:
                    ret = h_df['Close'].pct_change().rename(c)
                    df_list.append(ret)
            if df_list:
                returns_df = pd.concat(df_list, axis=1).dropna(how='all').fillna(0)
                cov_matrix = returns_df.cov() * 252 
            else:
                returns_df, cov_matrix = pd.DataFrame(), pd.DataFrame()
            
            pool_df = final_df[(final_df['sharpe'] > 0) & (final_df['mdd'] >= -0.25)].copy()
            if len(pool_df) == 0: pool_df = final_df.head(10).copy()
                
            def classify_quant_bucket(r):
                s = r.get('sharpe', 0)
                b = r.get('beta', 1.0)
                v = r.get('volatility', 1.0)
                if pd.isna(b): b = 1.0
                if s > 1.0 and b >= 1.0: return "🚀 成長型資產"
                elif s > 1.0 and b < 1.0: return "🛡️ 防禦型資產"
                elif s > 0 and v < 0.25: return "⚖️ 中性資產"
                else: return "🟡 其他觀察"
            
            pool_df['戰略定位'] = pool_df.apply(classify_quant_bucket, axis=1)
            
            grow_pool = pool_df[pool_df['戰略定位'] == "🚀 成長型資產"]
            def_pool = pool_df[pool_df['戰略定位'] == "🛡️ 防禦型資產"]
            neu_pool = pool_df[pool_df['戰略定位'] == "⚖️ 中性資產"]
            
            final_codes = greedy_mpt_optimization(grow_pool, returns_df, target_n=4, metric_col='sharpe', target_vol_max=0.15)
            final_codes = greedy_mpt_optimization(def_pool, returns_df, target_n=3, metric_col='sharpe', initial_selected=final_codes, target_vol_max=0.15)
            final_codes = greedy_mpt_optimization(neu_pool, returns_df, target_n=3, metric_col='sharpe', initial_selected=final_codes, target_vol_max=0.15)
            
            if len(final_codes) < 10:
                remaining_pool = pool_df[~pool_df['代號'].isin(final_codes)]
                extra_codes = greedy_mpt_optimization(remaining_pool, returns_df, target_n=10-len(final_codes), metric_col='sharpe', initial_selected=final_codes, target_vol_max=0.15)
                final_codes.extend(extra_codes)
                
            port_df = pool_df[pool_df['代號'].isin(final_codes)].sort_values(by=['戰略定位', 'sharpe'], ascending=[False, False])
            
            if len(port_df) > 0:
                c1, c2, c3 = st.columns([1.35, 0.9, 1.25])
                with c1:
                    def get_suggested_weight(row):
                        tag = row.get('戰略定位', '')
                        s = row.get('sharpe', 0)
                        v = row.get('volatility', 1.0)
                        if tag == "🚀 成長型資產": return "12% ~ 15%" if (s > 2.0 and v < 0.3) else "8% ~ 12%"
                        elif tag == "🛡️ 防禦型資產": return "8% ~ 12%"
                        else: return "5% ~ 8%"
                        
                    port_display = port_df[['代號', '名稱', '戰略定位', 'sharpe', 'volatility', 'beta']].copy()
                    port_display['建議權重'] = port_df.apply(get_suggested_weight, axis=1)
                    st.dataframe(
                        port_display, hide_index=True, use_container_width=True,
                        column_config={
                            "戰略定位": st.column_config.TextColumn("戰略配置定位"),
                            "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"),
                            "volatility": st.column_config.NumberColumn("波動率", format="%.2f"),
                            "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                        }
                    )
                    
                with c2:
                    def get_numeric_weight(row):
                        tag = row.get('戰略定位', '')
                        s = row.get('sharpe', 0)
                        v = row.get('volatility', 1.0)
                        if tag == "🚀 成長型資產": return 0.135 if (s > 2.0 and v < 0.3) else 0.10
                        elif tag == "🛡️ 防禦型資產": return 0.10
                        else: return 0.065
                        
                    avg_sharpe = port_df['sharpe'].mean()
                    avg_vol = port_df['volatility'].mean() * 100
                    avg_beta = port_df['beta'].mean()
                    
                    port_codes = port_df['代號'].tolist()
                    true_vol = avg_vol 
                    if not cov_matrix.empty and len(port_codes) > 1:
                        valid_codes = [c for c in port_codes if c in cov_matrix.columns]
                        if len(valid_codes) == len(port_codes):
                            raw_w = np.array(port_df.apply(get_numeric_weight, axis=1))
                            weights = raw_w / np.sum(raw_w)
                            port_variance = np.dot(weights.T, np.dot(cov_matrix.loc[valid_codes, valid_codes], weights))
                            true_vol = np.sqrt(port_variance) * 100
                    
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
                        <div style='font-size:0.8rem; color:#64748B; margin-bottom:12px;'>* 已依建議權重加權共變異數</div>
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
                        valid_codes = [c for c in port_codes if c in returns_df.columns]
                        if len(valid_codes) > 1:
                            corr_matrix_port = returns_df[valid_codes].corr()
                            short_codes = [c.replace('.TW', '').replace('.TWO', '') for c in valid_codes]
                            corr_matrix_port.columns = short_codes
                            corr_matrix_port.index = short_codes
                            
                            st.markdown("<div style='font-size:0.85rem; color:#9CA3AF; margin-bottom:5px;'>🔍 提示：滑鼠移至圖表右上角，點擊「⤢」可全螢幕放大</div>", unsafe_allow_html=True)
                            fig_corr = px.imshow(corr_matrix_port, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="組合資產關聯熱力圖")
                            fig_corr.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='#9CA3AF', size=11), margin=dict(t=40, b=20, l=20, r=20), height=380,
                                coloraxis_showscale=False
                            )
                            fig_corr.update_xaxes(type='category')
                            fig_corr.update_yaxes(type='category')
                            st.plotly_chart(fig_corr, use_container_width=True)

        else:
            st.subheader("💼 系統嚴選：【巴菲特護城河】長期集中價值配置")
            st.caption("依據巴菲特價值投資哲學：MDD < -25% 無情剔除！無視市場短期波動與 Beta，專注於企業長期經濟護城河。嚴選具備「高 ROE、高毛利、充沛自由現金流」的優質資產。")
            
            pool_df = final_df[(final_df['roe'] > 0.15) & (final_df['gross_margins'] > 40) & (final_df['mdd'] >= -0.25)].copy()
            if len(pool_df) < 10: pool_df = final_df[final_df['mdd'] >= -0.25].nlargest(10, 'Score').copy()
            
            def classify_buffett(r):
                g = r.get('eps_growth', 0)
                if pd.isna(g): g = 0
                return "🚀 成長護城河 (高EPS增長)" if g > 15.0 else "💰 穩健價值 (高ROE收息)"
            
            pool_df['戰略定位'] = pool_df.apply(classify_buffett, axis=1)
            port_df = pool_df.sort_values(by='Score', ascending=False).head(10)
            port_df = port_df.sort_values(by=['戰略定位', 'roe'], ascending=[False, False])

            if len(port_df) > 0:
                c1, c2, c3 = st.columns([1.35, 0.9, 1.25])
                with c1:
                    port_display = port_df[['代號', '名稱', '戰略定位', 'roe', 'gross_margins', 'fcf_yield', 'pe']].copy()
                    port_display['建議權重'] = f"{100.0/len(port_df):.1f}%"
                    
                    st.dataframe(
                        port_display, hide_index=True, use_container_width=True,
                        column_config={
                            "戰略定位": st.column_config.TextColumn("戰略配置定位"),
                            "roe": st.column_config.NumberColumn("ROE", format="%.2f"),
                            "gross_margins": st.column_config.NumberColumn("毛利率", format="%.2f"),
                            "fcf_yield": st.column_config.NumberColumn("FCF收益", format="%.2f"),
                            "pe": st.column_config.NumberColumn("本益比", format="%.2f"),
                        }
                    )
                    
                with c2:
                    avg_roe = port_df['roe'].mean() * 100
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
                    moat_df['roe'] = moat_df['roe'] * 100
                    moat_df['short_name'] = moat_df['代號'].astype(str).str.replace('.TW', '').str.replace('.TWO', '')
                    
                    fig_bar = go.Figure(data=[
                        go.Bar(name='ROE (%)', x=moat_df['short_name'], y=moat_df['roe'], marker_color='#10B981'),
                        go.Bar(name='毛利率 (%)', x=moat_df['short_name'], y=moat_df['gross_margins'], marker_color='#3B82F6')
                    ])
                    fig_bar.update_layout(
                        title="組合企業護城河指標 (Quality Metrics)",
                        barmode='group',
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                        font=dict(color='#9CA3AF', size=11), margin=dict(t=40, b=20, l=20, r=20), height=300,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_bar.update_xaxes(type='category', showgrid=False, linecolor='#334155')
                    fig_bar.update_yaxes(showgrid=True, gridcolor='#1E293B')
                    st.plotly_chart(fig_bar, use_container_width=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側設定篩選維度，並點擊「啟動終端運算」。")
