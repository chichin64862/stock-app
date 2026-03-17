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

# --- 1. 介面設定 (專業名稱) ---
st.set_page_config(
    page_title="台股量化與價值分析終端", 
    page_icon="📊", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 專業儀表板風格 (俐落三色系：深藍/專業藍/灰白) ---
st.markdown("""
<style>
    /* 全域背景：深海藍 */
    .stApp { background-color: #0B0F19 !important; }
    [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1F2937; }
    h1, h2, h3, p, span, div, label { color: #F3F4F6 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* 選單樣式：專業極簡 */
    div[role="listbox"] ul { background-color: #1F2937 !important; border: 1px solid #374151; }
    li[role="option"] { color: #F3F4F6 !important; background-color: #1F2937 !important; }
    li[role="option"]:hover { background-color: #2563EB !important; color: white !important; }
    input { background-color: #111827 !important; color: white !important; border: 1px solid #374151 !important; border-radius: 4px !important; }
    
    /* 股票卡片：方正俐落 */
    .stock-card { 
        background-color: #111827; padding: 20px; border-radius: 6px; 
        border: 1px solid #1F2937; margin-bottom: 20px; 
        border-left: 4px solid #2563EB; /* 專業藍色側邊條 */
    }
    .card-header {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #1F2937; padding-bottom: 12px; margin-bottom: 15px;
    }
    .header-title { font-size: 1.5rem; font-weight: 600; color: #FFFFFF; letter-spacing: 0.5px; }
    .header-price { font-size: 1.2rem; color: #9CA3AF; margin-left: 12px; font-family: 'Courier New', Courier, monospace;}
    
    /* 標籤：收斂色彩，微圓角 */
    .tag { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; margin-left: 6px; letter-spacing: 0.5px; }
    .tag-logic { background-color: #1E3A8A; color: #BFDBFE; border: 1px solid #1D4ED8; }
    .tag-sector { background-color: #374151; color: #E5E7EB; border: 1px solid #4B5563; }
    .tag-warn { background-color: #7F1D1D; color: #FECACA; border: 1px solid #991B1B; }
    .tag-quality { background-color: #064E3B; color: #BBF7D0; border: 1px solid #065F46; }
    
    /* 數據網格 */
    .metrics-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
        background-color: #1F2937; padding: 1px; border-radius: 4px; border: 1px solid #374151;
    }
    .metric-item { 
        display: flex; flex-direction: column; align-items: flex-start; justify-content: center; 
        background-color: #111827; padding: 12px;
    }
    .m-label { color: #9CA3AF; font-size: 0.8rem; margin-bottom: 4px; text-transform: uppercase; }
    .m-val { color: #FFFFFF; font-weight: bold; font-size: 1.1rem; font-family: 'Courier New', monospace; }
    .m-high { color: #34D399; } .m-warn { color: #F87171; }
    
    /* Reverse DCF 區塊 */
    .dcf-box {
        background-color: #111827; border: 1px solid #374151; border-left: 4px solid #8B5CF6;
        padding: 12px 15px; margin-top: 15px; border-radius: 4px;
    }
    
    /* AI 區塊 */
    .ai-box {
        background-color: #111827; border: 1px solid #1F2937; border-left: 4px solid #3B82F6;
        padding: 15px; margin-top: 15px; border-radius: 4px;
        font-size: 0.95rem; line-height: 1.6; color: #D1D5DB;
    }
    
    /* 按鈕：專業藍色系 */
    .stDownloadButton button, .stButton button { 
        background-color: #1F2937 !important; border: 1px solid #374151 !important; 
        color: #F3F4F6 !important; border-radius: 4px !important; font-weight: 600 !important;
    }
    .stDownloadButton button:hover, .stButton button:hover { 
        border-color: #3B82F6 !important; color: #3B82F6 !important; background-color: #111827 !important;
    }
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
    st.error("系統偵測不到 API Key，部分 AI 功能可能受限。")

# --- 5. 字型下載與註冊 ---
@st.cache_resource
def setup_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    urls = [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC-Regular.ttf",
        "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosanstc/NotoSansTC-Regular.ttf",
        "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.ttf"
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
                except:
                    continue
        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
        return 'ChineseFont'
    except Exception as e:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont('MSung-Light'))
            return 'MSung-Light'
        except:
            return 'Helvetica'

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

# --- 8. 評分邏輯 ---
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

    for idx, row in calc_df.iterrows():
        total_score = 0
        total_weight = 0
        
        if logic_type == "Quant":
            config = {
                'Sharpe': {'col': 'sharpe', 'dir': 'max', 'w': 3.0, 'cat': '動能'},
                'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 2.0, 'cat': '風險'},
                'Beta': {'col': 'beta', 'dir': 'mid', 'w': 1.0, 'cat': '風險'}, 
            }
        else:
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
            if sh > 1.0: q_tag = "高 CP 值"
            elif sh < 0: q_tag = "低動能"
            
            if sh > 1.0 and vol < 0.3: plans.append("防禦型買進")
            elif sh > 0.5 and b > 1.2: plans.append("積極型買進")
            elif vol > 0.5: plans.append("高波動警示")
            else: plans.append("中立觀望")
            
        else: 
            roe = row.get('roe', 0)
            gm = row.get('gross_margins', 0)
            de = row.get('de_ratio', 100)
            fcf = row.get('fcf_yield', 0)
            
            if roe > 15 and gm > 40 and de < 100 and fcf > 0: q_tag = "護城河"
            elif de > 200 or fcf < -5: q_tag = "財務風險"
            
            if q_tag == "護城河" and final > 70: plans.append("價值浮現")
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#374151'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=30, r=30), height=250, font=dict(color='#D1D5DB')
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
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='Price', line=dict(color='#60A5FA', width=2)))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['MA60'], name='MA60', line=dict(color='#FCD34D', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=[history_df.index[-1]], y=[current_price], mode='markers', marker=dict(color='#34D399', size=8), showlegend=False))

    fig.update_layout(
        title=dict(text=f"<b>中長期趨勢研判</b><br><span style='font-size:12px; color:#9CA3AF'>季線乖離 {bias_pct:.1f}% | {status_text}</span>", font=dict(color='#F3F4F6', size=14), y=0.95),
        xaxis=dict(showgrid=False, linecolor='#374151', tickfont=dict(color='#6B7280')),
        yaxis=dict(showgrid=True, gridcolor='#1F2937', tickfont=dict(color='#6B7280')),
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
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"分析服務暫時無回應 (代碼: {r.status_code})"
    except Exception as e:
        return "分析服務連線逾時，請稍後再試。"

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
    
    safe_name = str(stock_data.get('名稱', '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    safe_code = str(stock_data.get('代號', '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    safe_strat = str(stock_data.get('Strategy', '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    story.append(Paragraph(f"分析標的: {safe_name} ({safe_code})", h2_style))
    story.append(Paragraph(f"系統判定: {safe_strat}", normal_style))
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
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F2937')),
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
        clean_text = stock_data['ai_analysis'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('**', '').replace('##', '')
        for line in clean_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, normal_style))
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
1. 量化風險：夏普值=[SHARPE], 波動率=[VOL]%, Beta=[BETA].
2. 價值護城河：ROE=[ROE]%, 毛利率=[GM]%, FCF收益率=[FCF_Y]%, 負債權益比=[DE]%, EPS成長=[EPS_G]%.
3. 估值與時機：PE=[PE], Reverse DCF 隱含成長=[IMPLIED_G]%, 季線乖離=[MA_BIAS]%.
(註：若標的為ETF，財報數據如ROE、毛利率為 N/A 屬正常現象，請專注於量化風險與時機分析)

【輸出要求】
請依照「[LOGIC_NAME]」模型視角，輸出三大分析模塊：
1. **核心檢驗**：針對該模型最看重的指標進行評判。
2. **估值與風險**：檢視 Reverse DCF 與潛在財務或波動風險。
3. **操作結論**：結合中長期均線乖離狀況，給出客觀且具體的行動建議。
"""

# --- 11. 主程式 ---
with st.sidebar:
    st.title("⚙️ 終端控制面板")
    
    st.subheader("分析核心模型切換")
    logic_choice_tw = st.radio(
        "選擇運算引擎", 
        ["價值護城河 (高ROE/毛利/現金流)", "量化風控模型 (夏普值/波動率/Beta)"],
        index=0, label_visibility="collapsed"
    )
    st.session_state['current_logic'] = "Quant" if "量化" in logic_choice_tw else "Buffett"
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader("📂 擴充資料源 (CSV/Excel)", type=['csv','xlsx'], accept_multiple_files=True)
    if uploaded_files: 
        st.session_state['tej_data'] = process_tej_upload(uploaded_files)
        st.success(f"已成功掛載 {len(uploaded_files)} 份外部數據庫")
    
    st.markdown("---")
    st.subheader("資料池篩選")
    
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
        with st.spinner(f"正在執行矩陣運算，擷取 {len(target_stocks)} 檔標的數據..."):
            raw, hist_map = batch_scan_stocks(target_stocks, st.session_state['tej_data'])
            raw = sanitize_data(raw)
            st.session_state['raw_data'] = raw
            st.session_state['history_storage'] = hist_map 
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 台股量化與價值分析終端")
    logic_badge = "量化風控引擎" if st.session_state['current_logic'] == "Quant" else "價值護城河引擎"
    st.caption(f"STATUS: ONLINE | ENGINE: **{logic_badge}** | MODULES: Portfolio, AI Insight, PDF Export")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    current_logic = st.session_state['current_logic']
    
    if df.empty:
        st.error("檢索結果為空，請確認資料源連線狀態或代碼有效性。")
    else:
        final_df, df_norm = calculate_score(df, logic_type=current_logic)
        
        st.subheader("🏆 系統評級與排名")
        st.dataframe(
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
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🎯 個股深度檢驗面板")
        
        def safe_num(val): return 0 if (pd.isna(val) or val is None) else val

        for idx, row in final_df.head(10).iterrows():
            code = row['代號']
            
            with st.container():
                industry_tag = f"<span class='tag tag-sector'>{row['industry']}</span>"
                
                q_tag = row['Quality']
                if q_tag in ["High CP", "護城河", "Quality"]: quality_tag = f"<span class='tag tag-quality'>💎 {q_tag}</span>"
                elif q_tag in ["Low CP", "財務風險", "Profitless"]: quality_tag = f"<span class='tag tag-warn'>⚠️ {q_tag}</span>"
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
                    st.markdown(f"""<div class='metrics-grid'>
<div class='metric-item'><span class='m-label'>夏普值 (Sharpe)</span><span class='m-val m-high'>{safe_num(row.get('sharpe')):.2f}</span></div>
<div class='metric-item'><span class='m-label'>波動率 (Volatility)</span><span class='m-val'>{safe_num(row.get('volatility'))*100:.1f}%</span></div>
<div class='metric-item'><span class='m-label'>Beta (對盤連動)</span><span class='m-val'>{safe_num(row.get('beta')):.2f}</span></div>
<div class='metric-item'><span class='m-label'>ROE (權益報酬)</span><span class='m-val m-high'>{"N/A" if pd.isna(row.get('roe')) else f"{safe_num(row.get('roe'))*100 if row.get('roe') and row.get('roe')<1 else safe_num(row.get('roe')):.1f}%"}</span></div>
<div class='metric-item'><span class='m-label'>毛利率 (Gross Margin)</span><span class='m-val m-high'>{"N/A" if pd.isna(row.get('gross_margins')) else f"{safe_num(row.get('gross_margins')):.1f}%"}</span></div>
<div class='metric-item'><span class='m-label'>EPS成長 (EPS YoY)</span><span class='m-val m-high'>{"N/A" if pd.isna(row.get('eps_growth')) else f"{safe_num(row.get('eps_growth')):.1f}%"}</span></div>
<div class='metric-item'><span class='m-label'>本益比 (P/E)</span><span class='m-val'>{"N/A" if pd.isna(row.get('pe')) else f"{safe_num(row.get('pe')):.2f}"}</span></div>
<div class='metric-item'><span class='m-label'>FCF 收益率</span><span class='m-val'>{"N/A" if pd.isna(row.get('fcf_yield')) else f"{safe_num(row.get('fcf_yield')):.2f}%"}</span></div>
<div class='metric-item'><span class='m-label'>季線乖離 (MA Bias)</span><span class='m-val'>{safe_num(row.get('priceToMA60'))*100:.1f}%</span></div>
</div>""", unsafe_allow_html=True)
                    
                    implied_g = row.get('implied_growth', np.nan)
                    actual_g = row.get('eps_growth', 0) / 100 
                    if not pd.isna(implied_g):
                        if implied_g > actual_g + 0.1: dcf_status = "🔴 預估過熱"
                        elif implied_g < actual_g - 0.05: dcf_status = "🟢 具安全邊際"
                        else: dcf_status = "🟡 估值公允"
                        st.markdown(f"""<div class='dcf-box'>
<div style='font-size: 0.85rem; color: #9CA3AF; font-weight: 600; text-transform: uppercase;'>Reverse DCF 估值檢驗 (r=10%, TV=2%)</div>
<div style='display: flex; justify-content: space-between; margin-top: 8px;'>
<span>市場隱含期望成長: <b style='color: #F3F4F6; font-size: 1.1rem;'>{implied_g*100:.1f}%</b></span>
<span style='font-weight: 600;'>{dcf_status}</span>
</div></div>""", unsafe_allow_html=True)
                    
                    b1, b2 = st.columns(2)
                    if b1.button(f"🧠 執行 AI 洞察", key=f"ai_{idx}"):
                        current_today = datetime.now().strftime('%Y年%m月%d日')
                        if current_logic == "Quant":
                            l_name = "量化風控模型"
                            l_desc = "優先考量夏普值(CP值)與波動率(風險)，並解釋Beta值。"
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
                    
                    pdf_payload = row.to_dict()
                    if code in st.session_state['ai_results']:
                        pdf_payload['ai_analysis'] = st.session_state['ai_results'][code]
                    
                    pdf = create_pdf(pdf_payload)
                    file_name_dl = f"{code} {row['名稱']} ({(row.get('full_symbol', code))})_Report.pdf"
                    b2.download_button("📥 輸出 PDF 報告", pdf, file_name_dl, key=f"dl_{idx}")

                with c3:
                    if code in hist_storage and not hist_storage[code].empty:
                        st.plotly_chart(plot_trend_dashboard(row['名稱'], hist_storage[code], row.get('priceToMA60', 0)), use_container_width=True)
                    else:
                        st.warning("無 K 線數據")

                if code in st.session_state['ai_results']:
                    st.markdown(f"<div class='ai-box'>{st.session_state['ai_results'][code]}</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # 【核心新增】模型專屬投資組合 (Model-Driven Portfolio)
        st.markdown("---")
        st.subheader("💼 系統嚴選：模型專屬投資組合 (Model-Driven Portfolio)")
        
        portfolio_df = final_df.head(10).copy()
        
        if len(portfolio_df) > 0:
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.markdown(f"**核心驅動：{logic_badge}**")
                st.markdown("依據當前運算模型，由系統在您的篩選池中優化出的**等權重 (Equal-Weight)** 投資組合清單（最多 10 檔）：")
                
                port_display = portfolio_df[['代號', '名稱', 'industry', 'Score']].copy()
                port_display['配置權重'] = "10.0%" if len(portfolio_df) >= 10 else f"{100.0/len(portfolio_df):.1f}%"
                st.dataframe(port_display, hide_index=True, use_container_width=True)
                
            with c2:
                fig_pie = px.pie(portfolio_df, names='industry', title="板塊曝險分佈 (Sector Exposure)", hole=0.4, color_discrete_sequence=px.colors.sequential.Tealgrn)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#D1D5DB'))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # 組合整體面貌 (Aggregate Metrics)
            avg_pe = portfolio_df['pe'].mean()
            avg_yield = portfolio_df['yield'].mean()
            
            if current_logic == "Quant":
                avg_sharpe = portfolio_df['sharpe'].mean()
                avg_vol = portfolio_df['volatility'].mean() * 100
                st.markdown(f"""<div class='metrics-grid' style='grid-template-columns: repeat(4, 1fr);'>
<div class='metric-item'><span class='m-label'>組合平均本益比</span><span class='m-val'>{'N/A' if pd.isna(avg_pe) else f"{avg_pe:.2f}"}</span></div>
<div class='metric-item'><span class='m-label'>組合平均殖利率</span><span class='m-val'>{'N/A' if pd.isna(avg_yield) else f"{avg_yield:.2f}%"}</span></div>
<div class='metric-item'><span class='m-label'>組合平均夏普值</span><span class='m-val m-high'>{'N/A' if pd.isna(avg_sharpe) else f"{avg_sharpe:.2f}"}</span></div>
<div class='metric-item'><span class='m-label'>組合平均波動率</span><span class='m-val'>{'N/A' if pd.isna(avg_vol) else f"{avg_vol:.1f}%"}</span></div>
</div>""", unsafe_allow_html=True)
            else:
                avg_roe = portfolio_df['roe'].mean() * 100
                avg_fcf = portfolio_df['fcf_yield'].mean()
                st.markdown(f"""<div class='metrics-grid' style='grid-template-columns: repeat(4, 1fr);'>
<div class='metric-item'><span class='m-label'>組合平均本益比</span><span class='m-val'>{'N/A' if pd.isna(avg_pe) else f"{avg_pe:.2f}"}</span></div>
<div class='metric-item'><span class='m-label'>組合平均殖利率</span><span class='m-val'>{'N/A' if pd.isna(avg_yield) else f"{avg_yield:.2f}%"}</span></div>
<div class='metric-item'><span class='m-label'>組合平均 ROE</span><span class='m-val m-high'>{'N/A' if pd.isna(avg_roe) else f"{avg_roe:.1f}%"}</span></div>
<div class='metric-item'><span class='m-label'>組合平均 FCF 收益率</span><span class='m-val m-high'>{'N/A' if pd.isna(avg_fcf) else f"{avg_fcf:.2f}%"}</span></div>
</div>""", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側設定分析參數，並點擊「啟動終端運算」。")
