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
import io
import os
import time
from datetime import datetime
import urllib.request
import html

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
    
    .stButton button, .stDownloadButton button { 
        background-color: #1E293B !important; border: 1px solid #334155 !important; 
        color: #F8FAFC !important; border-radius: 2px !important; font-weight: 600 !important;
    }
    .stButton button:hover, .stDownloadButton button:hover { 
        border-color: #3B82F6 !important; color: #3B82F6 !important; background-color: #0D131F !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 初始化 ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None
if 'history_storage' not in st.session_state: st.session_state['history_storage'] = {}
if 'ai_results' not in st.session_state: st.session_state['ai_results'] = {}
if 'current_logic' not in st.session_state: st.session_state['current_logic'] = "Quant" 
if 'panel_page' not in st.session_state: st.session_state['panel_page'] = 1 

# --- 4. API Key ---
try: api_key = st.secrets["GEMINI_API_KEY"]
except Exception: st.error("系統偵測不到 API Key，AI 洞察功能將受限。")

# --- 5. 字型下載與註冊 (強固 PDF 輸出) ---
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

# --- 6. 核心數據引擎 ---
def create_resilient_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_tw_stock_list():
    try:
        import twstock
        codes = twstock.codes
        stock_map = {}
        industry_map = {}
        for code, info in codes.items():
            if info.type in ['股票', 'ETF']:
                suffix = '.TW' if info.market == '上市' else '.TWO'
                full = f"{code}{suffix}"
                stock_map[full] = f"{full} {info.name}"
                group = info.group if info.group else info.type
                if group not in industry_map: industry_map[group] = []
                industry_map[group].append(full)
        return stock_map, industry_map
    except: return {}, {}

stock_map, industry_map = get_tw_stock_list()

def get_stock_data(symbol):
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): symbol += '.TW'
        ticker = yf.Ticker(symbol)
        try: info = ticker.info 
        except: info = {}
        try:
            hist = ticker.history(period="6mo")
            if not hist.empty and hist['Volume'].iloc[-1] == 0: pass 
        except: hist = pd.DataFrame()

        def g(k): return info.get(k)

        data = {
            'close_price': g('currentPrice') or g('previousClose'),
            'pe': g('trailingPE'),
            'peg': g('pegRatio'),
            'pb': g('priceToBook'),
            'rev_growth': g('revenueGrowth'),
            'eps_growth': g('earningsGrowth'),
            'trailing_eps': g('trailingEps'), 
            'gross_margins': g('grossMargins'),
            'yield': g('dividendYield'),
            'roe': g('returnOnEquity'),
            'beta': g('beta'),
            'market_cap': g('marketCap'),
            'fcf': g('freeCashflow'),
            'debt_to_equity': g('debtToEquity'),
            'sector': g('sector') or 'General',
            'history': hist
        }
        return data
    except Exception: return None

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

def process_tej_upload(uploaded_files):
    if not uploaded_files: return None
    tej_map = {}
    if not isinstance(uploaded_files, list): uploaded_files = [uploaded_files]
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            df.columns = [str(c).strip() for c in df.columns]
            code_col = next((c for c in df.columns if '代號' in c or 'Code' in c), None)
            if not code_col: continue 
            for _, row in df.iterrows():
                raw_code = str(row[code_col]).split('.')[0].strip()
                if raw_code in tej_map: tej_map[raw_code].update(row.to_dict())
                else: tej_map[raw_code] = row.to_dict()
        except: continue
    return tej_map

# --- 7. 批量掃描 ---
@st.cache_data(ttl=3600, show_spinner=False)
def batch_scan_stocks(stock_list, tej_data=None):
    results = []
    history_map = {} 
    RISK_FREE_RATE = 0.015 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_stock = {executor.submit(get_stock_data, s.split(' ')[0]): s for s in stock_list}
        
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            try:
                code = stock_str.split(' ')[0].split('.')[0]
                name = code 
                
                if len(stock_str.split(' ')) > 1: name = stock_str.split(' ')[1]
                else:
                    try:
                        import twstock
                        if code in twstock.codes: name = twstock.codes[code].name
                    except: pass

                y_data = future.result()
                if y_data is None: continue

                price = np.nan; pe = np.nan; pb = np.nan; dy = np.nan
                rev_growth = np.nan; eps_growth = np.nan; margins = np.nan
                fcf_yield = np.nan; de_ratio = np.nan; beta_val = np.nan
                peg = np.nan; roe = np.nan; volatility = np.nan; sharpe = np.nan; implied_g = np.nan
                chips = 0; ma_bias = 0

                if y_data:
                    hist = y_data.get('history')
                    if hist is not None and not hist.empty:
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
                    
                    if pd.isna(price): price = y_data.get('close_price')
                    
                    def get_val(key):
                        v = y_data.get(key)
                        return float(v) if v is not None else np.nan

                    pe = get_val('pe'); pb = get_val('pb'); roe = get_val('roe')
                    raw_dy = get_val('yield')
                    if not pd.isna(raw_dy): dy = raw_dy * 100 
                    raw_rev = get_val('rev_growth')
                    if not pd.isna(raw_rev): rev_growth = raw_rev * 100
                    raw_eps = get_val('eps_growth')
                    if not pd.isna(raw_eps): eps_growth = raw_eps * 100
                    raw_margin = get_val('gross_margins')
                    if not pd.isna(raw_margin): margins = raw_margin * 100
                    peg = get_val('peg')
                    beta_val = get_val('beta')
                    de_ratio = get_val('debt_to_equity')
                    
                    m_cap = get_val('market_cap')
                    fcf = get_val('fcf')
                    if not pd.isna(m_cap) and not pd.isna(fcf) and m_cap > 0:
                        fcf_yield = (fcf / m_cap) * 100
                    
                    t_eps = get_val('trailing_eps')
                    if pd.isna(t_eps) and not pd.isna(pe) and pe > 0 and not pd.isna(price):
                        t_eps = price / pe
                    if not pd.isna(price) and not pd.isna(t_eps):
                        implied_g = calculate_implied_growth(price, t_eps)
                
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    for k, v in t_row.items():
                        if '法人' in k or 'Chips' in k: chips = float(v) if v != '-' else 0

                industry = 'General'
                if code in ['2330', '2454', '2303', '3034', '3035', '2379', '2382', '3231']: industry = 'Semicon'
                elif code.startswith('28') or code in ['5880']: industry = 'Finance'
                elif code in ['1101', '1301', '2002', '2603', '1802', '1605']: industry = 'Cyclical'
                elif code.startswith('00'): industry = 'ETF' 

                if not pd.isna(price):
                    results.append({
                        '代號': code, '名稱': name, 'close_price': price,
                        'pe': pe, 'pb': pb, 'yield': dy, 'roe': roe,
                        'rev_growth': rev_growth, 'eps_growth': eps_growth, 'gross_margins': margins,
                        'fcf_yield': fcf_yield, 'de_ratio': de_ratio, 'beta': beta_val,
                        'sharpe': sharpe, 'implied_growth': implied_g,
                        'peg': peg, 'chips': chips,
                        'volatility': volatility, 'priceToMA60': ma_bias,
                        'industry': industry,
                        'full_symbol': stock_str
                    })
            except: continue
    
    df = pd.DataFrame(results)
    cols = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'roe', 'rev_growth', 'eps_growth', 'gross_margins', 'fcf_yield', 'de_ratio', 'beta', 'sharpe', 'implied_growth', 'peg', 'chips', 'volatility', 'priceToMA60', 'industry']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df, history_map

# --- 8. 評分邏輯 (完全吻合紀律長贏理論) ---
def calculate_score(df, logic_type="Quant"):
    if df.empty: return df, None
    
    required_cols = ['pe', 'pb', 'yield', 'rev_growth', 'eps_growth', 'gross_margins', 'fcf_yield', 'de_ratio', 'beta', 'sharpe', 'implied_growth', 'peg', 'volatility', 'roe', 'priceToMA60']
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
            b = row.get('beta', 1)
            
            # 【核心正名】：嚴格落實林哲群教授理論
            if sh > 1.0 and b >= 1.0: q_tag = "高品質成長"
            elif sh > 1.0 and b < 1.0: q_tag = "風險優化防禦"
            elif sh < 0: q_tag = "風險報酬不對等"
            else: q_tag = "中立觀望"
            
            if sh > 1.0: plans.append("納入效率前緣")
            elif sh < 0: plans.append("避開標的")
            else: plans.append("中立觀望")
            
        else: 
            roe = row.get('roe', 0)
            gm = row.get('gross_margins', 0)
            de = row.get('de_ratio', 100)
            fcf = row.get('fcf_yield', 0)
            
            if roe > 15 and gm > 40 and de < 100 and fcf > 0: q_tag = "護城河優良"
            elif de > 200 or fcf < -5: q_tag = "財務風險"
            else: q_tag = "中立觀望"
            
            if q_tag == "護城河優良" and final > 70: plans.append("價值浮現")
            elif q_tag == "財務風險": plans.append("避開標的")
            elif final > 60: plans.append("合理估值")
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
        ['Beta (風險係數)', safe_str(stock_data.get('beta'), "{:.2f}"), '季線乖離 (MA Bias)', safe_str(stock_data.get('priceToMA60')*100 if not pd.isna(stock_data.get('priceToMA60')) else np.nan, "{:.1f}%")],
        ['ROE (權益報酬)', safe_str(stock_data.get('roe')*100 if not pd.isna(stock_data.get('roe')) else np.nan, "{:.1f}%"), '毛利率 (Gross Margin)', safe_str(stock_data.get('gross_margins'), "{:.1f}%")],
        ['營收 YoY (Rev Growth)', safe_str(stock_data.get('rev_growth'), "{:.1f}%"), 'EPS YoY (EPS Growth)', safe_str(stock_data.get('eps_growth'), "{:.1f}%")],
        ['本益比 (P/E)', safe_str(stock_data.get('pe')), 'FCF 收益率 (FCF Yield)', safe_str(stock_data.get('fcf_yield'), "{:.2f}%")],
        ['負債權益比 (D/E)', safe_str(stock_data.get('de_ratio'), "{:.1f}%"), '隱含成長率 (Implied G)', safe_str(stock_data.get('implied_growth')*100 if not pd.isna(stock_data.get('implied_growth')) else np.nan, "{:.1f}%")]
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

# 【核心更新】AI Prompt 注入《紀律長贏》理論
AI_PROMPT_TEMPLATE = """
請扮演專業的法人機構量化研究員，使用**繁體中文 (Traditional Chinese)** 分析 [STOCK] ([SECTOR])。
【時間基準】今天是 [CURRENT_DATE]，請以最新視角撰寫，絕對不要在報告中捏造或顯示過去的歷史日期。

【驅動模型】：[LOGIC_NAME]
[LOGIC_DESC]

【量化與財務數據】
1. 量化風險：夏普值=[SHARPE], 波動率=[VOL]%, Beta=[BETA].
2. 價值護城河：ROE=[ROE]%, 毛利率=[GM]%, FCF收益率=[FCF_Y]%, 負債權益比=[DE]%, EPS成長=[EPS_G]%.
3. 估值與時機：PE=[PE], Reverse DCF 隱含成長=[IMPLIED_G]%, 季線乖離=[MA_BIAS]%.
(註：若標的為ETF，財報數據如ROE、毛利率為 N/A 屬正常現象，請專注於量化風險與時機分析)

【輸出要求】
請依照「[LOGIC_NAME]」模型視角，輸出三大分析模塊：
1. **風險與效率前緣檢驗**：依據清大林哲群教授《紀律長贏》的學理，嚴格檢視夏普值是否匹配風險（若為負值請強烈警示風險報酬不對等）。解釋高 Beta 且高夏普為何屬於「高品質成長」，而非單純投機。
2. **護城河與估值**：檢視財務體質，並對照 Reverse DCF 評估是否透支未來。
3. **操作結論**：結合 Beta 屬性與季線乖離，給出客觀的資產配置定位與具體行動建議。
"""

# --- 11. 標籤解析函數 (依據林哲群教授理論精確詮釋) ---
def get_tag_explanation(row, logic_type):
    tag = row.get('Quality', '')
    sharpe = row.get('sharpe', 0)
    roe = (row.get('roe', 0)*100) if not pd.isna(row.get('roe')) else 0
    gm = row.get('gross_margins', 0)
    fcf = row.get('fcf_yield', 0)
    de = row.get('de_ratio', 0)
    
    if logic_type == "Quant":
        if tag == "高品質成長": return f"**為什麼被系統評定為「🔥 高品質成長」？**<br>此標的近期年化夏普值高達 **{sharpe:.2f}**。依據林哲群教授《紀律長贏》的『風險優化』觀念：雖然其股價波動度與 Beta 值高於大盤，但其報酬率的累積速度遠高於風險的增加，使得風險調整後的投資績效極具吸引力，位於效率前緣上，絕非單純投機動能。"
        elif tag == "風險優化防禦": return f"**為什麼被系統評定為「🛡️ 風險優化防禦」？**<br>此標的近期年化夏普值高達 **{sharpe:.2f}**，且 Beta 值小於 1。這代表投資人每承擔 1 單位的波動風險，就能獲得超過 1 單位的超額報酬，同時具備低於大盤的波動特性，是優化投資組合風險的防禦基石。"
        elif tag == "風險報酬不對等": return f"**為什麼被系統評定為「⚠️ 風險報酬不對等」？**<br>此標的夏普值為 **{sharpe:.2f}**（小於 0）。如《紀律長贏》所述：即便部分資產可能擁有高殖利率或熱門題材，但當夏普率為負時，代表其實際累積報酬低於無風險利率。投資此類資產並未為投資人帶來與風險相匹配的回報，不符合長期紀律投資精神。"
        else: return "**標籤說明**<br>該標的各項量化指標落在市場均值區間，風險與報酬呈現中性，未觸發極端強勢或弱勢的演算法警示。"
    else:
        if tag == "護城河優良": return f"**為什麼被系統評定為「護城河優良」？**<br>此標的完美符合巴菲特的核心護城河條件：<br>1. **高資本效率**：ROE 達 **{roe:.1f}%** (標準 > 15%)。<br>2. **強大定價權**：毛利率達 **{gm:.1f}%** (標準 > 40%)。<br>3. **真實變現力**：自由現金流收益率為正且無過度舉債。這是一檔擁有深厚經濟壁壘的優質資產。"
        elif tag == "財務風險": return f"**為什麼被系統評定為「財務風險」？**<br>系統偵測到其存在潛在危機：負債權益比(D/E) 高達 **{de:.1f}%**，或自由現金流收益率低至 **{fcf:.2f}%**。這代表該公司可能正在靠過度舉債擴張，或賺到的盈餘無法轉換為真實現金，具有高度「虛胖」風險。"
        else: return "**標籤說明**<br>該標的在財務體質（ROE、毛利率、現金流）上表現中規中矩，雖未達嚴苛的「絕對護城河」標準，但也無明顯的財務致命傷。"

# --- 12. 演算法核心：MPT 貪婪降維挑選法 ---
def greedy_correlation_selection(pool_df, returns_df, target_n, metric_col, initial_selected=None):
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
        best_score = -9999
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
            obj_score = norm_metric - (avg_corr * 0.8) 
            
            if obj_score > best_score:
                best_score = obj_score
                best_code = code
        
        if best_code:
            selected_codes.append(best_code)
            candidates.remove(best_code)
        else:
            break
            
    return selected_codes

# --- 13. 主程式介面 ---
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
        with st.spinner(f"正在擷取並運算 {len(target_stocks)} 檔標的數據..."):
            raw, hist_map = batch_scan_stocks(target_stocks, st.session_state['tej_data'])
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
    st.caption(f"STATUS: ONLINE | ENGINE: **{logic_badge}** | ALGO: 清大林哲群《紀律長贏》框架導入")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    current_logic = st.session_state['current_logic']
    
    if df.empty:
        st.error("檢索結果為空，請確認資料源連線狀態或代碼有效性。")
    else:
        final_df, df_norm = calculate_score(df, logic_type=current_logic)
        
        st.subheader("🏆 終端檢索清單")
        st.caption("💡 點擊下方表格內的任意資料可查看「深度解析」。**(若表格呈現白底，請點擊網頁右上角「⋮」-> Settings -> Theme 選擇 Dark)**")
        
        df_event = st.dataframe(
            final_df[['代號', '名稱', 'industry', 'Score', 'Quality', 'Strategy', 'sharpe', 'roe', 'fcf_yield', 'implied_growth']],
            column_config={
                "industry": st.column_config.TextColumn("產業/板塊"),
                "Score": st.column_config.ProgressColumn("綜合評分", min_value=0, max_value=100, format="%.1f"),
                "Quality": st.column_config.TextColumn("系統標籤"),
                "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"),
                "roe": st.column_config.NumberColumn("ROE", format="%.1f"),
                "fcf_yield": st.column_config.NumberColumn("FCF收益", format="%.1f%%"),
                "implied_growth": st.column_config.NumberColumn("隱含成長", format="%.2f"),
            },
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row" 
        )
        
        st.markdown("---")
        
        selected_rows = df_event.selection.rows
        
        if selected_rows:
            sel_idx = selected_rows[0]
            sel_row = final_df.iloc[sel_idx]
            
            st.subheader(f"💡 系統標籤深度解析：{sel_row['名稱']} ({sel_row['代號']})")
            explanation = get_tag_explanation(sel_row, current_logic)
            st.markdown(f"<div class='explain-box'>{explanation}</div>", unsafe_allow_html=True)
            
            st.subheader("🎯 個股深度檢驗面板 (單檔檢視)")
            st.info("📌 目前為「單檔檢視模式」。若要恢復分頁瀏覽模式，請在上方清單中再次點擊以取消選取。")
            display_df = final_df.iloc[[sel_idx]]
            
        else:
            st.subheader("🎯 個股深度檢驗面板 (全覽分頁)")
            
            total_stocks = len(final_df)
            items_per_page = 5
            total_pages = max(1, (total_stocks - 1) // items_per_page + 1)
            
            if st.session_state['panel_page'] > total_pages:
                st.session_state['panel_page'] = total_pages
                
            start_idx = (st.session_state['panel_page'] - 1) * items_per_page
            end_idx = start_idx + items_per_page
            display_df = final_df.iloc[start_idx:end_idx]
            
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
                if q_tag in ["高品質成長", "風險優化防禦", "護城河優良", "Quality"]: quality_tag = f"<span class='tag tag-moat'>💎 {q_tag}</span>"
                elif q_tag in ["風險報酬不對等", "財務風險", "Profitless"]: quality_tag = f"<span class='tag tag-risk'>⚠️ {q_tag}</span>"
                else: quality_tag = ""
                    
                logic_tag = f"<span class='tag tag-logic'>{logic_badge}</span>"
                
                st.markdown(f"""<div class='stock-card'>
<div class='card-header'>
<div><span class='header-title'>{row['名稱']} ({code})</span><span class='header-price'>${row['close_price']}</span></div>
<div>{logic_tag}{industry_tag}{quality_tag}</div>
</div>""", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 1.8, 1.5])
                
                with c1:
                    if idx in df_norm.index:
                        st.plotly_chart(plot_radar_chart_ui(row['名稱'], get_radar_data(df_norm.loc[idx])), use_container_width=True)
                
                with c2:
                    sharpe_val = safe_num(row.get('sharpe'))
                    sh_color = "q-up" if sharpe_val > 1 else ("q-down" if sharpe_val < 0 else "")
                    rev_val = safe_num(row.get('rev_growth'))
                    eps_val = safe_num(row.get('eps_growth'))
                    
                    st.markdown(f"""<div class='quote-board'>
<div class='quote-item'><span class='q-label'>夏普值 (Sharpe)</span><span class='q-val {sh_color}'>{sharpe_val:.2f}</span></div>
<div class='quote-item'><span class='q-label'>波動率 (Volatility)</span><span class='q-val'>{safe_num(row.get('volatility'))*100:.1f}%</span></div>
<div class='quote-item'><span class='q-label'>Beta (風險係數)</span><span class='q-val'>{safe_num(row.get('beta')):.2f}</span></div>
<div class='quote-item'><span class='q-label'>ROE (權益報酬)</span><span class='q-val'>{"N/A" if pd.isna(row.get('roe')) else f"{safe_num(row.get('roe'))*100 if row.get('roe') and row.get('roe')<1 else safe_num(row.get('roe')):.1f}%"}</span></div>
<div class='quote-item'><span class='q-label'>毛利率 (Gross Mgn)</span><span class='q-val'>{"N/A" if pd.isna(row.get('gross_margins')) else f"{safe_num(row.get('gross_margins')):.1f}%"}</span></div>
<div class='quote-item'><span class='q-label'>本益比 (P/E)</span><span class='q-val'>{"N/A" if pd.isna(row.get('pe')) else f"{safe_num(row.get('pe')):.2f}"}</span></div>
<div class='quote-item'><span class='q-label'>營收 YoY</span><span class='q-val {get_color_class(rev_val)}'>{rev_val:.1f}%</span></div>
<div class='quote-item'><span class='q-label'>EPS YoY</span><span class='q-val {get_color_class(eps_val)}'>{eps_val:.1f}%</span></div>
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
                            if st.button(f"🧠 執行 AI 洞察", key=f"ai_{idx}"):
                                current_today = datetime.now().strftime('%Y年%m月%d日')
                                if current_logic == "Quant":
                                    l_name = "量化風控模型"
                                    l_desc = "請以清大林哲群教授《紀律長贏》的『風險優化』核心精神：辨識高夏普/高Beta的『高品質成長型資產』，以及高夏普/低Beta的『風險優化防禦資產』。並警示夏普值為負的風險報酬不對等現象。"
                                else:
                                    l_name = "價值護城河模型"
                                    l_desc = "優先考量ROE、毛利率、FCF現金流收益率，防禦高負債風險。"
        
                                p_txt = AI_PROMPT_TEMPLATE.replace("[LOGIC_NAME]", l_name) \
                                    .replace("[LOGIC_DESC]", l_desc) \
                                    .replace("[STOCK]", row['名稱']) \
                                    .replace("[SECTOR]", str(row['industry'])) \
                                    .replace("[CURRENT_DATE]", current_today) \
                                    .replace("[SHARPE]", str(safe_num(row.get('sharpe')))) \
                                    .replace("[VOL]", str(safe_num(row.get('volatility'))*100)) \
                                    .replace("[BETA]", str(safe_num(row.get('beta')))) \
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
                            if st.button(f"🔄 更新 AI 洞察", key=f"ai_re_{idx}"):
                                current_today = datetime.now().strftime('%Y年%m月%d日')
                                if current_logic == "Quant":
                                    l_name = "量化風控模型"
                                    l_desc = "請以清大林哲群教授《紀律長贏》的『風險優化』核心精神：辨識高夏普/高Beta的『高品質成長型資產』，以及高夏普/低Beta的『風險優化防禦資產』。並警示夏普值為負的風險報酬不對等現象。"
                                else:
                                    l_name = "價值護城河模型"
                                    l_desc = "優先考量ROE、毛利率、FCF現金流收益率，防禦高負債風險。"
        
                                p_txt = AI_PROMPT_TEMPLATE.replace("[LOGIC_NAME]", l_name) \
                                    .replace("[LOGIC_DESC]", l_desc) \
                                    .replace("[STOCK]", row['名稱']) \
                                    .replace("[SECTOR]", str(row['industry'])) \
                                    .replace("[CURRENT_DATE]", current_today) \
                                    .replace("[SHARPE]", str(safe_num(row.get('sharpe')))) \
                                    .replace("[VOL]", str(safe_num(row.get('volatility'))*100)) \
                                    .replace("[BETA]", str(safe_num(row.get('beta')))) \
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
                            st.download_button("📥 輸出 PDF 報告", pdf, file_name_dl, key=f"dl_{idx}")

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
                st.markdown(f"<div style='text-align: center; color: #9CA3AF; padding-top: 8px;'>第 <b>{st.session_state['panel_page']}</b> 頁 / 共 <b>{total_pages}</b> 頁 (總計 {total_stocks} 檔)</div>", unsafe_allow_html=True)
            with col_p3:
                if st.button("下一頁 ➡️", use_container_width=True, disabled=(st.session_state['panel_page'] == total_pages)):
                    st.session_state['panel_page'] += 1
                    st.rerun()

        # =========================================================
        # 💼 林哲群教授：紀律長贏資產配置模組 (馬可維茲關聯性優化)
        # =========================================================
        st.markdown("---")
        st.subheader("💼 【紀律長贏】效率前緣投資組合 (Markowitz MPT Optimization)")
        
        df_list = []
        for c, h_df in hist_storage.items():
            if not h_df.empty and 'Close' in h_df.columns:
                ret = h_df['Close'].pct_change().rename(c)
                df_list.append(ret)
        
        if df_list:
            returns_df = pd.concat(df_list, axis=1).dropna(how='all').fillna(0)
            cov_matrix = returns_df.cov() * 252 
        else:
            returns_df = pd.DataFrame()
            cov_matrix = pd.DataFrame()
        
        port_df = pd.DataFrame()
        
        if current_logic == "Quant":
            st.caption("依據《紀律長贏》核心精神：嚴格要求 **高夏普值 (>1.0)** 確保位於效率前緣。系統進一步透過 MPT 貪婪演算法尋找 **「相關性最低」** 的資產組合，為您配置 5 檔高品質成長 (高Beta) 與 5 檔風險優化防禦 (低Beta)。")
            
            # 【嚴格鎖死學理邏輯】
            pool_df = final_df[final_df['sharpe'] > 1.0].copy()
            if len(pool_df) < 10: pool_df = final_df[final_df['sharpe'] > 0].copy()
            if len(pool_df) == 0: pool_df = final_df.head(10).copy()
                
            def classify_quant(r):
                s = r.get('sharpe', 0)
                b = r.get('beta', 1.0)
                if pd.isna(b): b = 1.0
                
                # 嚴格落實林哲群教授定義：高夏普且高Beta = 高品質成長
                if s > 1.0 and b >= 1.0:
                    return "🔥 高品質成長 (高夏普/高Beta)"
                elif s > 1.0 and b < 1.0:
                    return "🛡️ 風險優化防禦 (高夏普/低Beta)"
                elif s > 0:
                    return "🟡 動能及格 (夏普>0)"
                else:
                    return "⚠️ 風險報酬不對等"
            
            pool_df['戰略定位'] = pool_df.apply(classify_quant, axis=1)
            
            # 將符合嚴格標準的資產分別放入水桶
            atk_pool = pool_df[pool_df['戰略定位'] == "🔥 高品質成長 (高夏普/高Beta)"]
            def_pool = pool_df[pool_df['戰略定位'] == "🛡️ 風險優化防禦 (高夏普/低Beta)"]
            
            atk_selected = greedy_correlation_selection(atk_pool, returns_df, target_n=5, metric_col='sharpe')
            def_selected = greedy_correlation_selection(def_pool, returns_df, target_n=5, metric_col='sharpe', initial_selected=atk_selected)
            
            final_codes = atk_selected + def_selected
            if len(final_codes) < 10:
                remaining_pool = pool_df[~pool_df['代號'].isin(final_codes)]
                extra_codes = greedy_correlation_selection(remaining_pool, returns_df, target_n=10-len(final_codes), metric_col='sharpe', initial_selected=final_codes)
                final_codes.extend(extra_codes)
                
            port_df = pool_df[pool_df['代號'].isin(final_codes)].sort_values(by=['戰略定位', 'sharpe'], ascending=[False, False])
            
        else:
            st.caption("依據《紀律長贏》與價值護城河理論：優先篩選 **ROE > 15%**，並透過相關性過濾，為您搭配兼具成長動能與價值收息，且走勢互補的資產組合。")
            
            pool_df = final_df[(final_df['roe'] > 0.15) & (final_df['gross_margins'] > 40)].copy()
            if len(pool_df) < 10: pool_df = final_df.nlargest(max(10, len(final_df)//2), 'roe').copy()
            if len(pool_df) == 0: pool_df = final_df.head(10).copy()
                
            def classify_buffett(r):
                g = r.get('eps_growth', 0)
                if pd.isna(g): g = 0
                return "🚀 成長護城河 (高EPS增長)" if g > 15.0 else "💰 穩健價值 (高ROE收息)"
            
            pool_df['戰略定位'] = pool_df.apply(classify_buffett, axis=1)
            grw_pool = pool_df[pool_df['戰略定位'] == "🚀 成長護城河 (高EPS增長)"]
            val_pool = pool_df[pool_df['戰略定位'] == "💰 穩健價值 (高ROE收息)"]
            
            grw_selected = greedy_correlation_selection(grw_pool, returns_df, target_n=5, metric_col='Score')
            val_selected = greedy_correlation_selection(val_pool, returns_df, target_n=5, metric_col='Score', initial_selected=grw_selected)
            
            final_codes = grw_selected + val_selected
            if len(final_codes) < 10:
                remaining_pool = pool_df[~pool_df['代號'].isin(final_codes)]
                extra_codes = greedy_correlation_selection(remaining_pool, returns_df, target_n=10-len(final_codes), metric_col='Score', initial_selected=final_codes)
                final_codes.extend(extra_codes)
                
            port_df = pool_df[pool_df['代號'].isin(final_codes)].sort_values(by=['戰略定位', 'roe'], ascending=[False, False])

        if len(port_df) > 0:
            c1, c2, c3 = st.columns([1.6, 1, 1])
            with c1:
                port_display = port_df[['代號', '名稱', '戰略定位', 'sharpe', 'volatility', 'beta', 'roe']].copy()
                port_display['配置權重'] = f"{100.0/len(port_df):.1f}%"
                
                st.dataframe(
                    port_display, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "戰略定位": st.column_config.TextColumn("戰略配置定位"),
                        "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"),
                        "volatility": st.column_config.NumberColumn("波動率", format="%.2f"),
                        "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                        "roe": st.column_config.NumberColumn("ROE", format="%.2f"),
                    }
                )
                
            with c2:
                avg_sharpe = port_df['sharpe'].mean()
                avg_vol = port_df['volatility'].mean() * 100
                avg_beta = port_df['beta'].mean()
                
                port_codes = port_df['代號'].tolist()
                true_vol = avg_vol 
                if not cov_matrix.empty and len(port_codes) > 1:
                    valid_codes = [c for c in port_codes if c in cov_matrix.columns]
                    if len(valid_codes) == len(port_codes):
                        weights = np.array([1/len(port_codes)] * len(port_codes))
                        port_variance = np.dot(weights.T, np.dot(cov_matrix.loc[valid_codes, valid_codes], weights))
                        true_vol = np.sqrt(port_variance) * 100
                
                st.markdown(f"""
                <div style='background-color:#1E293B; padding:15px; border-radius:4px; border:1px solid #334155;'>
                    <h4 style='margin-top:0; color:#F8FAFC; border-bottom:1px solid #334155; padding-bottom:10px;'>MPT 效率前緣檢驗</h4>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#9CA3AF;'>組合單檔平均波動率</span><span style='color:#F8FAFC; font-weight:bold; font-size:1.1rem;'>{avg_vol:.1f}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                        <span style='color:#F59E0B;'>MPT 真實組合波動率</span><span style='color:#10B981; font-weight:bold; font-size:1.3rem;'>{true_vol:.1f}%</span>
                    </div>
                    <div style='font-size:0.8rem; color:#64748B; margin-bottom:12px;'>* 經相關性抵銷後，真實風險顯著下降</div>
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
                        
                        fig_corr = px.imshow(corr_matrix_port, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="組合資產關聯熱力圖")
                        fig_corr.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='#9CA3AF', size=10), margin=dict(t=30, b=0, l=0, r=0), height=240,
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側設定篩選維度，並點擊「啟動終端運算」。")
