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
    page_title="熵值決策選股平台 (Dual Engine)", 
    page_icon="🦅", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 專業儀表板風格 ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    h1, h2, h3, p, span, div, label { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
    
    div[role="listbox"] ul { background-color: #262730 !important; }
    li[role="option"] { color: white !important; background-color: #262730 !important; }
    li[role="option"]:hover { background-color: #238636 !important; }
    input { background-color: #0d1117 !important; color: white !important; border: 1px solid #30363d !important; }
    
    .stock-card { 
        background-color: #1f2937; padding: 20px; border-radius: 12px; 
        border: 1px solid #374151; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .card-header {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #374151; padding-bottom: 12px; margin-bottom: 15px;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
    .header-price { font-size: 1.2rem; color: #9ca3af; margin-left: 10px; }
    
    .tag { padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; font-weight: bold; margin-left: 8px; }
    .tag-logic { background-color: #9333ea; color: white; border: 1px solid #a855f7; }
    .tag-sector { background-color: #3b82f6; color: white; border: 1px solid #2563eb; }
    .tag-warn { background-color: #b91c1c; color: white; border: 1px solid #ef4444; }
    .tag-quality { background-color: #7c3aed; color: white; border: 1px solid #8b5cf6; }
    
    /* 升級為 3x4 網格以容納夏普值與Beta */
    .metrics-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
        background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;
    }
    .metric-item { display: flex; flex-direction: column; align-items: flex-start; justify-content: center; }
    .m-label { color: #9ca3af; font-size: 0.85rem; margin-bottom: 4px; }
    .m-val { color: #ffffff; font-weight: bold; font-size: 1.1rem; font-family: 'Courier New', monospace; }
    .m-high { color: #4ade80; } .m-warn { color: #f87171; }
    
    .dcf-box {
        background-color: rgba(255, 215, 0, 0.05); border-left: 4px solid #FFD700;
        padding: 12px 15px; margin-top: 15px; border-radius: 4px;
    }
    
    .ai-box {
        background-color: #2d333b; border-left: 4px solid #58a6ff;
        padding: 15px; margin-top: 15px; border-radius: 4px;
        font-size: 0.95rem; line-height: 1.6; color: #e6e6e6;
    }
    
    .stDownloadButton button { background-color: #374151 !important; border: 1px solid #4b5563 !important; color: white !important; width: 100%; }
    .stDownloadButton button:hover { border-color: #60a5fa !important; color: #60a5fa !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. 初始化 ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None
if 'history_storage' not in st.session_state: st.session_state['history_storage'] = {}
if 'ai_results' not in st.session_state: st.session_state['ai_results'] = {}
if 'current_logic' not in st.session_state: st.session_state['current_logic'] = "Buffett"

# --- 4. API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！")

# --- 5. 字型下載與註冊 (保留防黑方塊修復) ---
@st.cache_resource
def setup_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    urls = [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC-Regular.ttf",
        "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosanstc/NotoSansTC-Regular.ttf"
    ]
    try:
        if not os.path.exists(font_path) or os.path.getsize(font_path) < 100000:
            for url in urls:
                try:
                    r = requests.get(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200 and len(r.content) > 100000:
                        with open(font_path, 'wb') as f: 
                            f.write(r.content)
                        break 
                except: continue
        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
        return 'ChineseFont'
    except Exception:
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
            if info.type == '股票':
                suffix = '.TW' if info.market == '上市' else '.TWO'
                full = f"{code}{suffix}"
                stock_map[full] = f"{full} {info.name}"
                if info.group not in industry_map: industry_map[info.group] = []
                industry_map[info.group].append(full)
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

# --- 7. 批量掃描 (加入夏普值計算) ---
@st.cache_data(ttl=3600, show_spinner=False)
def batch_scan_stocks(stock_list, tej_data=None):
    results = []
    history_map = {} 
    RISK_FREE_RATE = 0.015 # 台灣無風險利率假設 1.5%
    
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
                            
                            # 【新增】計算夏普值 (Sharpe Ratio)
                            ret_6m = (closes.iloc[-1] / closes.iloc[0]) - 1
                            ann_ret = ret_6m * 2 # 簡單年化
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

# --- 8. 評分邏輯 (雙引擎重構) ---
def calculate_score(df, logic_type="Buffett"):
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

    # 【核心】雙引擎評分權重定義
    for idx, row in calc_df.iterrows():
        total_score = 0
        total_weight = 0
        
        if logic_type == "Quant":
            # 1. 量化風控模型 (Quant Risk Model)
            config = {
                'Sharpe': {'col': 'sharpe', 'dir': 'max', 'w': 3.0, 'cat': '動能'},
                'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 2.0, 'cat': '風險'},
                'Beta': {'col': 'beta', 'dir': 'mid', 'w': 1.0, 'cat': '風險'}, # Beta 偏好適中(不極端)
            }
        else:
            # 2. 巴菲特護城河模型 (Buffett Value Model)
            config = {
                'ROE': {'col': 'roe', 'dir': 'max', 'w': 2.0, 'cat': '財報'},
                'GrossMargin': {'col': 'gross_margins', 'dir': 'max', 'w': 2.0, 'cat': '財報'},
                'FCF_Yield': {'col': 'fcf_yield', 'dir': 'max', 'w': 2.0, 'cat': '價值'},
                'DE_Ratio': {'col': 'de_ratio', 'dir': 'min', 'w': 2.0, 'cat': '風險'},
                'EPS_Growth': {'col': 'eps_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'},
            }
            
        for name, setting in config.items():
            val = row.get(setting['col'])
            all_vals = calc_df[setting['col']]
            rank = all_vals.rank(pct=True).get(idx, 0.5)
            
            if setting['dir'] == 'max': norm = rank
            elif setting['dir'] == 'min': norm = 1 - rank
            else: norm = 1 - abs(rank - 0.5)*2 # 越靠近中位數越好 (針對Beta)
                
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm * 100
            total_score += norm * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final, 1))
        
        # 產生動態標籤與策略
        q_tag = ""
        if logic_type == "Quant":
            sh = row.get('sharpe', 0)
            vol = row.get('volatility', 1)
            b = row.get('beta', 1)
            if sh > 1.0: q_tag = "High CP"
            elif sh < 0: q_tag = "Low CP"
            
            if sh > 1.0 and vol < 0.3: plans.append("🚀 高CP防禦佈局 (Buy)")
            elif sh > 0.5 and b > 1.2: plans.append("⚔️ 攻擊動能 (Momentum)")
            elif vol > 0.5: plans.append("⚠️ 高波動風險 (Warning)")
            else: plans.append("🟡 觀望 (Wait)")
            
        else: # Buffett Logic
            roe = row.get('roe', 0)
            gm = row.get('gross_margins', 0)
            de = row.get('de_ratio', 100)
            fcf = row.get('fcf_yield', 0)
            
            if roe > 15 and gm > 40 and de < 100 and fcf > 0: q_tag = "Moat"
            elif de > 200 or fcf < -5: q_tag = "Toxic Debt"
            
            if q_tag == "Moat" and final > 70: plans.append("💎 護城河買點 (Strong Buy)")
            elif q_tag == "Toxic Debt": plans.append("⛔ 負債/流失警示 (Avoid)")
            elif final > 60: plans.append("🟡 合理持有 (Hold)")
            else: plans.append("⛔ 觀望 (Wait)")
            
        quality_tags.append(q_tag)
            
    df['Score'] = scores
    df['Strategy'] = plans
    df['Quality'] = quality_tags
    return df.sort_values('Score', ascending=False), df_norm

# --- 9. 繪圖函數 ---
def get_radar_data(df_norm_row):
    cats = {'價值 (Value)': 0, '成長 (Growth)': 0, '動能 (Momentum)': 0, '風險 (Risk)': 0, '財報 (Financials)': 0}
    counts = {'價值 (Value)': 0, '成長 (Growth)': 0, '動能 (Momentum)': 0, '風險 (Risk)': 0, '財報 (Financials)': 0}
    map_dict = {'價值': '價值 (Value)', '成長': '成長 (Growth)', '動能': '動能 (Momentum)', '風險': '風險 (Risk)', '財報': '財報 (Financials)'}
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
        fill='toself', name=title, line_color='#00e676', fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#4b5563'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=30, r=30), height=250, font=dict(color='#e6e6e6')
    )
    return fig

def plot_trend_dashboard(title, history_df, ma_bias):
    if history_df is None or history_df.empty: return None
    history_df['MA60'] = history_df['Close'].rolling(window=60).mean()
    current_price = history_df['Close'].iloc[-1]
    bias_pct = ma_bias * 100
    if bias_pct > 15: status_text = f"🔴 留意過熱 (Overheated)"
    elif bias_pct > 5: status_text = f"🔥 動能強勢 (Strong)"
    elif bias_pct > -5: status_text = f"🟡 盤整持有 (Hold)"
    elif bias_pct > -15: status_text = f"🟢 超跌/價值 (Value Zone)"
    else: status_text = f"⛔ 趨勢轉空 (Avoid)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='Price', line=dict(color='#29b6f6', width=2.5)))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['MA60'], name='MA60', line=dict(color='#ffca28', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=[history_df.index[-1]], y=[current_price], mode='markers', marker=dict(color='#00e676', size=10), showlegend=False))

    fig.update_layout(
        title=dict(text=f"<b>配置時機判定 (Trend vs Value)</b><br><span style='font-size:14px; color:#e6e6e6'>{bias_pct:.1f}%  {status_text}</span>", font=dict(color='white', size=16), y=0.95),
        xaxis=dict(showgrid=False, linecolor='#4b5563', tickfont=dict(color='#9ca3af')),
        yaxis=dict(showgrid=True, gridcolor='#374151', tickfont=dict(color='#9ca3af')),
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
    if not api_key: return "⚠️ 未設定 API Key"
    if not st.session_state.get('ai_model_name'):
        st.session_state['ai_model_name'] = get_valid_model(api_key)
    
    target_model = st.session_state['ai_model_name']
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        session = create_resilient_session()
        r = session.post(url, headers=headers, json=data, timeout=60)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"❌ API 錯誤: {r.status_code}"
    except Exception as e:
        return f"❌ 連線例外: {str(e)}"

def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    font_name = font_name_global 
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=20, alignment=1, spaceAfter=20)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName=font_name, fontSize=14, spaceBefore=10, spaceAfter=10, textColor=colors.darkblue)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14)
    
    logic_name = "量化風控 (Quant Risk)" if st.session_state['current_logic'] == "Quant" else "巴菲特護城河 (Buffett Value)"
    story.append(Paragraph(f"熵值決策選股及AI深度分析報告 [{logic_name}]", title_style))
    story.append(Paragraph(f"生成時間 (Time): {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 10))
    
    safe_name = str(stock_data.get('名稱', '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    safe_code = str(stock_data.get('代號', '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    safe_strat = str(stock_data.get('Strategy', '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    story.append(Paragraph(f"標的 (Target): {safe_name} ({safe_code})", h2_style))
    story.append(Paragraph(f"戰略指令 (Strategy): {safe_strat}", normal_style))
    story.append(Spacer(1, 10))
    
    def safe_str(val, fmt="{:.2f}"):
        try: return "N/A" if (pd.isna(val) or val is None) else fmt.format(float(val))
        except: return "N/A"

    metrics_data = [
        ['夏普值 (Sharpe)', safe_str(stock_data.get('sharpe')), '波動率 (Volatility)', safe_str(stock_data.get('volatility')*100 if not pd.isna(stock_data.get('volatility')) else np.nan, "{:.1f}%")],
        ['ROE (%)', safe_str(stock_data.get('roe')*100 if not pd.isna(stock_data.get('roe')) else np.nan, "{:.2f}%"), '毛利率 (Gross Margin)', safe_str(stock_data.get('gross_margins'), "{:.2f}%")],
        ['負債權益比 (D/E)', safe_str(stock_data.get('de_ratio'), "{:.2f}%"), 'FCF 收益率', safe_str(stock_data.get('fcf_yield'), "{:.2f}%")],
        ['隱含成長率 (Implied G)', safe_str(stock_data.get('implied_growth')*100 if not pd.isna(stock_data.get('implied_growth')) else np.nan, "{:.2f}%"), 'Beta (風險係數)', safe_str(stock_data.get('beta'), "{:.2f}")]
    ]
    t = Table(metrics_data, colWidths=[120, 110, 120, 110])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
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
        story.append(Paragraph("AI 深度投資建議 (AI Investment Insights)", h2_style))
        clean_text = stock_data['ai_analysis'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('**', '').replace('##', '')
        for line in clean_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 5))
                
    try: doc.build(story)
    except Exception as e: print(e)
    buffer.seek(0)
    return buffer

# 【核心升級】動態 AI 提示詞模板
AI_PROMPT_TEMPLATE = """
請扮演華爾街基金經理人，使用**繁體中文 (Traditional Chinese)** 分析 [STOCK] ([SECTOR])。
【時間基準】今天是 [CURRENT_DATE]，請以最新視角撰寫，**絕對不要捏造過去日期**。

【用戶選擇的分析模型】：[LOGIC_NAME]
[LOGIC_DESC]

【3x4 數據矩陣】
1. 量化風險：夏普值=[SHARPE], 波動率=[VOL]%, Beta=[BETA].
2. 價值護城河：ROE=[ROE]%, 毛利率=[GM]%, FCF收益率=[FCF_Y]%, 負債權益比=[DE]%, EPS成長=[EPS_G]%.
3. 估值與時機：PE=[PE], Reverse DCF 隱含成長=[IMPLIED_G]%, 季線乖離=[MA_BIAS]%.

【輸出要求】
請依照用戶選擇的「[LOGIC_NAME]」模型視角，輸出三大模塊：
1. **核心檢驗 (Core Logic Test)**：針對該模型最看重的指標進行評判(量化看夏普/Beta；巴菲特看ROE/毛利/FCF)。
2. **估值與風險 (Valuation & Risks)**：檢視 Reverse DCF 與負債/波動風險。
3. **操作結論 (Action)**：結合季線乖離給出具體買賣建議。
排版清晰條理。
"""

# --- 11. 主程式 ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    
    # 【核心新增】選股邏輯切換器
    st.subheader("🧠 核心選股邏輯")
    logic_choice_tw = st.radio(
        "請選擇演算法核心", 
        ["👴 巴菲特護城河 (高ROE/高毛利/高FCF)", "📊 量化風控模型 (夏普值/低波動/Beta)"],
        index=0
    )
    st.session_state['current_logic'] = "Quant" if "量化" in logic_choice_tw else "Buffett"
    
    st.markdown("---")
    
    with st.expander("📂 匯入 TEJ (支援多檔)"):
        uploaded_files = st.file_uploader("上傳 CSV/Excel", type=['csv','xlsx'], accept_multiple_files=True)
        if uploaded_files: 
            st.session_state['tej_data'] = process_tej_upload(uploaded_files)
            st.success(f"已載入 TEJ 數據 (共 {len(uploaded_files)} 檔)")
    
    scan_mode = st.radio("模式選擇", ["🔥 熱門策略", "🏭 產業掃描", "⌨️ 自訂輸入"])
    
    strategies = {
        "🏆 台灣50 (TW50)": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2412.TW", "1301.TW"],
        "🤖 AI 伺服器 (AI Server)": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "2356.TW"],
        "💰 高股息殖利率 (High Yield)": ["2454.TW", "2303.TW", "2357.TW", "1101.TW", "2891.TW", "0056.TW"],
        "🍎 蘋果概念股 (Apple Concept)": ["2330.TW", "2317.TW", "3008.TW", "4938.TW", "2313.TW"],
        "🚗 電動車與車電 (EV/Auto)": ["2308.TW", "2317.TW", "6235.TW", "1536.TW", "5425.TW"],
        "🏦 金融保險 (Financials)": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "5880.TW", "2884.TW"],
        "🚢 傳產與航運 (Cyclical/Shipping)": ["2603.TW", "2609.TW", "2002.TW", "1301.TW", "1303.TW", "1605.TW"]
    }
    
    target_stocks = []
    if scan_mode == "⌨️ 自訂輸入":
        default = ["2330.TW 台積電", "2317.TW 鴻海", "2454.TW 聯發科", "2881.TW 富邦金"]
        options = sorted(list(stock_map.values())) if stock_map else default
        selected = st.multiselect("選擇股票", options, default=[s for s in default if s in options])
        manual = st.text_input("手動輸入代號", "")
        target_stocks = selected
        if manual: target_stocks.append(f"{manual}.TW")
        
    elif scan_mode == "🏭 產業掃描":
        ind_list = sorted(list(industry_map.keys()))
        ind = st.selectbox("選擇產業", ind_list)
        if ind: target_stocks = [stock_map[c] for c in industry_map[ind] if c in stock_map]
    else:
        strat_name = st.selectbox("策略集", list(strategies.keys()))
        target_stocks = strategies[strat_name]

    if st.button("🚀 啟動全自動掃描", type="primary"):
        st.session_state['scan_finished'] = False
        with st.spinner(f"正在以 [{st.session_state['current_logic']}] 模型進行矩陣運算..."):
            raw, hist_map = batch_scan_stocks(target_stocks, st.session_state['tej_data'])
            raw = sanitize_data(raw)
            st.session_state['raw_data'] = raw
            st.session_state['history_storage'] = hist_map 
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 40.0")
    logic_badge = "📊 量化風控" if st.session_state['current_logic'] == "Quant" else "👴 價值護城河"
    st.caption(f"目前啟動引擎：**{logic_badge}** | PDF Font Guarantee + 3x4 Matrix")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    current_logic = st.session_state['current_logic']
    
    if df.empty:
        st.error("❌ 查無數據，請檢查代號或網路。")
    else:
        # 套用新的雙引擎計分邏輯
        final_df, df_norm = calculate_score(df, logic_type=current_logic)
        
        st.subheader("🏆 潛力標的排行")
        st.dataframe(
            final_df[['代號', '名稱', 'industry', 'Score', 'Quality', 'Strategy', 'sharpe', 'roe', 'fcf_yield', 'implied_growth']],
            column_config={
                "industry": st.column_config.TextColumn("產業"),
                "Score": st.column_config.ProgressColumn("系統分數", min_value=0, max_value=100, format="%.1f"),
                "Quality": st.column_config.TextColumn("屬性標籤"),
                "sharpe": st.column_config.NumberColumn("夏普值(Sharpe)", format="%.2f"),
                "roe": st.column_config.NumberColumn("ROE", format="%.1f"),
                "fcf_yield": st.column_config.NumberColumn("FCF收益", format="%.1f%%"),
                "implied_growth": st.column_config.NumberColumn("隱含成長", format="%.2f"),
            },
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🎯 深度戰略分析")
        
        def safe_num(val): return 0 if (pd.isna(val) or val is None) else val

        for idx, row in final_df.head(10).iterrows():
            code = row['代號']
            
            with st.container():
                industry_tag = f"<span class='tag tag-sector'>{row['industry']}</span>"
                
                # 依據模型給予不同標籤顏色
                q_tag = row['Quality']
                if q_tag in ["High CP", "Moat", "Quality"]:
                    quality_tag = f"<span class='tag tag-quality'>💎 {q_tag}</span>"
                elif q_tag in ["Low CP", "Toxic Debt", "Profitless"]:
                    quality_tag = f"<span class='tag tag-warn'>⚠️ {q_tag}</span>"
                else:
                    quality_tag = ""
                    
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
                    # 【升級】3x4 數據矩陣 (加入夏普、波動、Beta、ROE)
                    st.markdown(f"""<div class='metrics-grid'>
<div class='metric-item'><span class='m-label'>夏普值 (Sharpe)</span><span class='m-val m-high'>{safe_num(row.get('sharpe')):.2f}</span></div>
<div class='metric-item'><span class='m-label'>波動率 (Volatility)</span><span class='m-val'>{safe_num(row.get('volatility'))*100:.1f}%</span></div>
<div class='metric-item'><span class='m-label'>Beta (對盤連動)</span><span class='m-val'>{safe_num(row.get('beta')):.2f}</span></div>
<div class='metric-item'><span class='m-label'>ROE (權益報酬)</span><span class='m-val m-high'>{safe_num(row.get('roe'))*100 if row.get('roe') and row.get('roe')<1 else safe_num(row.get('roe')):.1f}%</span></div>
<div class='metric-item'><span class='m-label'>毛利率 (Gross Margin)</span><span class='m-val m-high'>{safe_num(row.get('gross_margins')):.1f}%</span></div>
<div class='metric-item'><span class='m-label'>EPS成長 (EPS YoY)</span><span class='m-val m-high'>{safe_num(row.get('eps_growth')):.1f}%</span></div>
<div class='metric-item'><span class='m-label'>本益比 (P/E)</span><span class='m-val'>{safe_num(row.get('pe')):.2f}</span></div>
<div class='metric-item'><span class='m-label'>FCF 收益率</span><span class='m-val'>{safe_num(row.get('fcf_yield')):.2f}%</span></div>
<div class='metric-item'><span class='m-label'>負債權益比 (D/E)</span><span class='m-val'>{safe_num(row.get('de_ratio')):.1f}%</span></div>
</div>""", unsafe_allow_html=True)
                    
                    implied_g = row.get('implied_growth', np.nan)
                    actual_g = row.get('eps_growth', 0) / 100 
                    if not pd.isna(implied_g):
                        if implied_g > actual_g + 0.1: dcf_status = "🔴 預期過高 (Overpriced)"
                        elif implied_g < actual_g - 0.05: dcf_status = "🟢 安全邊際 (Margin of Safety)"
                        else: dcf_status = "🟡 估值合理 (Fairly Priced)"
                        st.markdown(f"""<div class='dcf-box'>
<div style='font-size: 0.85rem; color: #9ca3af; font-weight: bold;'>Reverse DCF 估值檢驗 (r=10%, TV=2%)</div>
<div style='display: flex; justify-content: space-between; margin-top: 8px;'>
<span>市場隱含成長率: <b style='color: white; font-size: 1.1rem;'>{implied_g*100:.1f}%</b></span>
<span style='font-weight: bold;'>{dcf_status}</span>
</div></div>""", unsafe_allow_html=True)
                    
                    b1, b2 = st.columns(2)
                    if b1.button(f"✨ {logic_badge} AI 分析", key=f"ai_{idx}"):
                        current_today = datetime.now().strftime('%Y年%m月%d日')
                        
                        # 動態寫入邏輯敘述
                        if current_logic == "Quant":
                            l_name = "量化風控模型 (Quant Risk Model)"
                            l_desc = "請優先看重『夏普值(大於1為佳)』確保CP值，其次看『波動率(越低越好)』控制絕對風險，並解釋『Beta值』代表的市場曝險程度。"
                        else:
                            l_name = "巴菲特價值模型 (Buffett Value Model)"
                            l_desc = "請優先看重『ROE(>15%)』、『毛利率(>40%)』確認護城河，嚴格檢查『FCF收益率(必須為正)』與『負債比』確保財務穩健。"

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
                    
                    pdf_payload = row.to_dict()
                    if code in st.session_state['ai_results']:
                        pdf_payload['ai_analysis'] = st.session_state['ai_results'][code]
                    
                    pdf = create_pdf(pdf_payload)
                    file_name_dl = f"{code} {row['名稱']} ({(row.get('full_symbol', code))})_Report.pdf"
                    b2.download_button("📥 下載報告 (Download PDF)", pdf, file_name_dl, key=f"dl_{idx}")

                with c3:
                    if code in hist_storage and not hist_storage[code].empty:
                        st.plotly_chart(plot_trend_dashboard(row['名稱'], hist_storage[code], row.get('priceToMA60', 0)), use_container_width=True)
                    else:
                        st.warning("無 K 線數據")

                if code in st.session_state['ai_results']:
                    st.markdown(f"<div class='ai-box'>{st.session_state['ai_results'][code]}</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動全自動掃描」開始。")
