import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import requests
import io
import os
from datetime import datetime

# PDF 函式庫
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="熵值決策選股平台 (Bilingual & Full Market)", 
    page_icon="🦅", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 專業儀表板風格 ---
st.markdown("""
<style>
    /* 全域深色 */
    .stApp { background-color: #0e1117 !important; }
    
    /* 側邊欄 */
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    
    /* 文字顏色 */
    h1, h2, h3, p, span, div, label { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
    
    /* 下拉選單修正 */
    div[role="listbox"] ul { background-color: #262730 !important; }
    li[role="option"] { color: white !important; background-color: #262730 !important; }
    li[role="option"]:hover { background-color: #238636 !important; }
    input { background-color: #0d1117 !important; color: white !important; border: 1px solid #30363d !important; }
    
    /* 【核心】專業戰略卡片 */
    .stock-card { 
        background-color: #1f2937; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #374151; 
        margin-bottom: 25px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* 卡片標題列 */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #374151;
        padding-bottom: 12px;
        margin-bottom: 15px;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
    .header-price { font-size: 1.2rem; color: #9ca3af; margin-left: 10px; }
    
    /* 標籤 */
    .tag { padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; font-weight: bold; margin-left: 8px; }
    .tag-strategy { background-color: #238636; color: white; border: 1px solid #2ea043; }
    .tag-buffett { background-color: #FFD700; color: black; border: 1px solid #b39700; }
    .tag-sector { background-color: #3b82f6; color: white; border: 1px solid #2563eb; }
    .tag-warn { background-color: #b91c1c; color: white; border: 1px solid #ef4444; }
    .tag-quality { background-color: #7c3aed; color: white; border: 1px solid #8b5cf6; }
    
    /* 中間數據網格 */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        background-color: rgba(0,0,0,0.2);
        padding: 15px;
        border-radius: 8px;
    }
    .metric-item { display: flex; justify-content: space-between; align-items: center; }
    .m-label { color: #9ca3af; font-size: 0.9rem; }
    .m-val { color: #ffffff; font-weight: bold; font-size: 1.0rem; font-family: 'Courier New', monospace; }
    .m-high { color: #4ade80; } 
    .m-warn { color: #f87171; }
    
    /* AI 分析區塊 */
    .ai-box {
        background-color: #2d333b;
        border-left: 4px solid #58a6ff;
        padding: 15px;
        margin-top: 15px;
        border-radius: 4px;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e6e6e6;
    }
    
    /* 下載按鈕 */
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

# --- 4. API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！")
    st.stop()

# --- 5. 字型下載與註冊 ---
@st.cache_resource
def setup_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    url = "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get(url, allow_redirects=True, timeout=10)
            if r.status_code == 200:
                with open(font_path, 'wb') as f: f.write(r.content)
        except: return False
    try:
        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
        return True
    except: return False

font_ready = setup_chinese_font()

# --- 6. 數據引擎 (全市場覆蓋) ---
def get_tw_stock_list():
    try:
        import twstock
        codes = twstock.codes
        stock_map = {}
        industry_map = {}
        # 【核心】遍歷所有代號，區分上市與上櫃
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
        # 自動補後綴 (雙重保險)
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): 
            # 預設先試 TW，若清單中有定義則用清單的
            symbol += '.TW'
            
        ticker = yf.Ticker(symbol)
        info = ticker.info 
        hist = ticker.history(period="6mo")
        
        data = {
            'close_price': info.get('currentPrice') or info.get('previousClose'),
            'pe': info.get('trailingPE'),
            'peg': info.get('pegRatio'),
            'pb': info.get('priceToBook'),
            'rev_growth': info.get('revenueGrowth'),
            'eps_growth': info.get('earningsGrowth'),
            'gross_margins': info.get('grossMargins'),
            'yield': info.get('dividendYield'),
            'roe': info.get('returnOnEquity'),
            'beta': info.get('beta'),
            'sector': info.get('sector', 'General'),
            'history': hist
        }
        return data
    except: return None

def calculate_synthetic_peg(pe, growth_rate):
    if pe and growth_rate and growth_rate > 0:
        return pe / (growth_rate * 100)
    return None

def sanitize_data(df):
    if df.empty: return df
    if 'yield' in df.columns:
        df['yield'] = df['yield'].apply(lambda x: x/100 if x > 20 else x)
    return df

# 支援多檔匯入
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
@st.cache_data(ttl=300, show_spinner=False)
def batch_scan_stocks(stock_list, tej_data=None):
    results = []
    history_map = {} 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_data, s.split(' ')[0]): s for s in stock_list}
        
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            try:
                code = stock_str.split(' ')[0].split('.')[0]
                name = stock_str.split(' ')[1] if len(stock_str.split(' ')) > 1 else code
                y_data = future.result()
                
                price = np.nan; pe = np.nan; pb = np.nan; dy = np.nan
                rev_growth = np.nan; eps_growth = np.nan; margins = np.nan
                peg = np.nan; roe = np.nan; volatility = 0.5
                chips = 0; ma_bias = 0

                if y_data:
                    # K線
                    hist = y_data.get('history')
                    if hist is not None and not hist.empty:
                        history_map[code] = hist 
                        closes = hist['Close']
                        if len(closes) > 10:
                            price = float(closes.iloc[-1])
                            volatility = closes.pct_change().std() * (252**0.5)
                            ma60 = closes.rolling(60).mean().iloc[-1]
                            if not pd.isna(ma60): ma_bias = (price / ma60) - 1
                    
                    if pd.isna(price): price = y_data.get('close_price')
                    pe = y_data.get('pe')
                    pb = y_data.get('pb')
                    roe = y_data.get('roe')
                    raw_dy = y_data.get('yield')
                    if raw_dy: dy = raw_dy * 100 
                    
                    raw_rev = y_data.get('rev_growth')
                    if raw_rev: rev_growth = raw_rev * 100
                    
                    raw_eps = y_data.get('eps_growth')
                    if raw_eps: eps_growth = raw_eps * 100
                    
                    raw_margin = y_data.get('gross_margins')
                    if raw_margin: margins = raw_margin * 100
                    
                    peg = y_data.get('peg')
                
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    for k, v in t_row.items():
                        if '法人' in k or 'Chips' in k: chips = float(v) if v != '-' else 0

                if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
                    peg = calculate_synthetic_peg(pe, rev_growth/100)

                industry = 'General'
                # 簡單產業映射 (可擴充)
                if code in ['2330', '2454', '2303', '3034', '3035', '2379', '2382', '3231']: industry = 'Semicon'
                elif code.startswith('28') or code in ['5880']: industry = 'Finance'
                elif code in ['1101', '1301', '2002', '2603', '1802', '1605']: industry = 'Cyclical'

                if not pd.isna(price):
                    results.append({
                        '代號': code, '名稱': name, 'close_price': price,
                        'pe': pe, 'pb': pb, 'yield': dy, 'roe': roe,
                        'rev_growth': rev_growth, 'eps_growth': eps_growth, 'gross_margins': margins,
                        'peg': peg, 'chips': chips,
                        'volatility': volatility, 'priceToMA60': ma_bias,
                        'industry': industry
                    })
            except: continue
    
    df = pd.DataFrame(results)
    # Auto-Heal Columns
    cols = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'roe', 'rev_growth', 'eps_growth', 'gross_margins', 'peg', 'chips', 'volatility', 'priceToMA60', 'industry']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
        
    return df, history_map

# --- 8. 評分邏輯 ---
def get_sector_config(industry):
    config = {
        'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 1, 'cat': '風險'}, 
    }
    if industry == 'Semicon': 
        config.update({
            'PEG': {'col': 'peg', 'dir': 'min', 'w': 1.5, 'cat': '成長'},
            'Rev Growth': {'col': 'rev_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'},
            'EPS Growth': {'col': 'eps_growth', 'dir': 'max', 'w': 1.5, 'cat': '成長'},
            'P/E': {'col': 'pe', 'dir': 'min', 'w': 1.0, 'cat': '價值'},
        })
    elif industry == 'Finance': 
        config.update({
            'Yield': {'col': 'yield', 'dir': 'max', 'w': 2.0, 'cat': '價值'},
            'P/B': {'col': 'pb', 'dir': 'min', 'w': 1.5, 'cat': '價值'},
            'ROE': {'col': 'roe', 'dir': 'max', 'w': 1.0, 'cat': '財報'},
        })
    else: 
        config.update({
            'P/E': {'col': 'pe', 'dir': 'min', 'w': 1.5, 'cat': '價值'},
            'Rev Growth': {'col': 'rev_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'},
            'EPS Growth': {'col': 'eps_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'},
            'Yield': {'col': 'yield', 'dir': 'max', 'w': 1.0, 'cat': '財報'},
        })
    return config

def check_buffett_criteria(row):
    roe = row.get('roe', 0)
    vol = row.get('volatility', 1.0)
    pe = row.get('pe', 100)
    if roe and roe < 1: roe = roe * 100 
    if pd.isna(roe): roe = 0
    score = 0
    if roe > 15: score += 1
    if vol < 0.35: score += 1
    if pe < 20 and pe > 0: score += 1
    return score >= 2

def calculate_score(df, use_buffett=False):
    if df.empty: return df, None
    
    required_cols = ['pe', 'pb', 'yield', 'rev_growth', 'eps_growth', 'gross_margins', 'peg', 'volatility', 'roe', 'priceToMA60']
    for col in required_cols:
        if col not in df.columns: df[col] = np.nan
    
    df_norm = df.copy()
    scores = []
    plans = []
    buffett_tags = []
    quality_tags = []
    
    fill_map = {c: 0 for c in required_cols}
    fill_map['pe'] = 50; fill_map['peg'] = 5; fill_map['volatility'] = 0.5
    calc_df = df.fillna(fill_map)

    for idx, row in calc_df.iterrows():
        config = get_sector_config(row.get('industry', 'General'))
        total_score = 0; total_weight = 0
        
        for name, setting in config.items():
            val = row.get(setting['col'])
            all_vals = calc_df[setting['col']]
            rank = all_vals.rank(pct=True).get(idx, 0.5)
            if setting['dir'] == 'max': norm = rank
            else: norm = 1 - rank
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm * 100
            total_score += norm * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        is_buffett = check_buffett_criteria(row)
        buffett_tags.append("🏅" if is_buffett else "")
        if use_buffett and is_buffett: final = min(100, final + 15)
        
        scores.append(round(final, 1))
        
        rev = row.get('rev_growth', 0); eps = row.get('eps_growth', 0); ma = row.get('priceToMA60', 0)
        q_tag = ""
        if rev > 20 and eps < 0: q_tag = "Profitless"
        elif rev > 15 and eps > 15: q_tag = "Quality"
        quality_tags.append(q_tag)
        
        if q_tag == "Profitless": plans.append("⚠️ 虛胖警告 (Profitless)")
        elif final > 75 and q_tag == "Quality": plans.append("💎 高品質爆發 (Quality Buy)")
        elif final > 75: plans.append("🚀 爆發成長 (Buy)")
        elif final > 60: plans.append("🟡 穩健持有 (Hold)")
        elif ma < -0.1: plans.append("🟢 超跌反彈 (Rebound)")
        elif ma > 0.2: plans.append("🔴 過熱拉回 (Overheated)")
        else: plans.append("⛔ 觀望 (Wait)")
            
    df['Score'] = scores
    df['Strategy'] = plans
    df['Buffett'] = buffett_tags
    df['Quality'] = quality_tags
    return df.sort_values('Score', ascending=False), df_norm

# --- 9. 繪圖函數 (中英對照) ---
def get_radar_data(df_norm_row):
    # 中英對照
    cats = {'價值 (Value)': 0, '成長 (Growth)': 0, '動能 (Momentum)': 0, '風險 (Risk)': 0, '財報 (Financials)': 0}
    counts = {'價值 (Value)': 0, '成長 (Growth)': 0, '動能 (Momentum)': 0, '風險 (Risk)': 0, '財報 (Financials)': 0}
    
    # 簡易映射
    map_dict = {'價值': '價值 (Value)', '成長': '成長 (Growth)', '動能': '動能 (Momentum)', '風險': '風險 (Risk)', '財報': '財報 (Financials)'}
    
    for col in df_norm_row.index:
        if str(col).endswith('_n'):
            cat_raw = str(col).split('_')[0]
            cat = map_dict.get(cat_raw, cat_raw) # 映射到中英
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
        r = requests.post(url, headers=headers, json=data, timeout=60)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"❌ API 錯誤: {r.status_code}"
    except Exception as e:
        return f"❌ 連線例外: {str(e)}"

# 【核心更新】中英對照 PDF
def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    font_name = 'ChineseFont' if font_ready else 'Helvetica'
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=20, alignment=1, spaceAfter=20)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName=font_name, fontSize=14, spaceBefore=10, spaceAfter=10, textColor=colors.darkblue)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14)
    
    story.append(Paragraph(f"熵值決策選股及AI深度分析報告 (Analysis Report)", title_style))
    story.append(Paragraph(f"生成時間 (Time): {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(f"標的 (Target): {stock_data['名稱']} ({stock_data['代號']})", h2_style))
    story.append(Paragraph(f"戰略指令 (Strategy): {stock_data['Strategy']}", normal_style))
    
    rev_g = stock_data.get('rev_growth', 0); eps_g = stock_data.get('eps_growth', 0)
    if rev_g > 20 and eps_g < 0:
        story.append(Paragraph(f"⚠️ 警告 (Warning): 檢測到虛胖成長 (Profitless Growth)", normal_style))
    story.append(Spacer(1, 10))
    
    # 中英對照數據表
    metrics_data = [
        ['收盤價 (Price)', f"{stock_data['close_price']}", '熵值分數 (Score)', f"{stock_data.get('Score', 'N/A')}"],
        ['本益比 (P/E)', f"{stock_data.get('pe', 'N/A')}", 'PEG Ratio', f"{stock_data.get('peg', 'N/A')}"],
        ['營收成長 (Rev Growth)', f"{stock_data.get('rev_growth', 0):.2f}%", 'EPS 成長 (EPS Growth)', f"{stock_data.get('eps_growth', 0):.2f}%"],
        ['毛利率 (Gross Margin)', f"{stock_data.get('gross_margins', 0):.2f}%", '殖利率 (Yield)', f"{stock_data.get('yield', 0):.2f}%"],
        ['波動率 (Volatility)', f"{stock_data.get('volatility', 0)*100:.1f}%", '季線乖離 (MA Bias)', f"{stock_data.get('priceToMA60', 0)*100:.1f}%"]
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
        clean_text = stock_data['ai_analysis'].replace('**', '').replace('##', '')
        for line in clean_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 5))
                
    try: doc.build(story)
    except Exception as e: print(e)
    buffer.seek(0)
    return buffer

AI_PROMPT = """
請扮演華爾街基金經理人，使用**繁體中文 (Traditional Chinese)** 分析 [STOCK] ([SECTOR])。
數據：PE=[PE], PEG=[PEG], 營收成長=[REV]%, EPS成長=[EPS_G]%, 毛利率=[GM]%, ROE=[ROE]%.
重點：
1. **成長品質**：營收與EPS是否同步成長？是否存在「虛胖」(營收增但EPS減)？
2. **估值風險**：PEG是否合理？
3. **結論**：給出操作建議。
請務必使用**繁體中文**回答。
"""

# --- 11. 主程式 ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    
    with st.expander("📂 匯入 TEJ (支援多檔)"):
        uploaded_files = st.file_uploader("上傳 CSV/Excel", type=['csv','xlsx'], accept_multiple_files=True)
        if uploaded_files: 
            st.session_state['tej_data'] = process_tej_upload(uploaded_files)
            st.success(f"已載入 TEJ 數據 (共 {len(uploaded_files)} 檔)")

    use_buffett = st.checkbox("🎩 啟用巴菲特選股", value=False)
    
    scan_mode = st.radio("模式選擇", ["🔥 熱門策略", "🏭 產業掃描", "⌨️ 自訂輸入"])
    
    # 策略選單 (中英對照)
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
        # 預設清單擴充
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
        with st.spinner("正在挖掘 Yahoo 數據 (含 EPS/毛利率 深度財報)..."):
            raw, hist_map = batch_scan_stocks(target_stocks, st.session_state['tej_data'])
            raw = sanitize_data(raw)
            st.session_state['raw_data'] = raw
            st.session_state['history_storage'] = hist_map 
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 36.0")
    st.caption("Bilingual UI + Full Market Coverage + Pro Report")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    
    if df.empty:
        st.error("❌ 查無數據，請檢查代號或網路。")
    else:
        final_df, df_norm = calculate_score(df, use_buffett)
        
        st.subheader("🏆 潛力標的排行")
        # 中英對照表頭
        st.dataframe(
            final_df[['代號', '名稱', 'industry', 'Score', 'Buffett', 'Quality', 'Strategy', 'rev_growth', 'eps_growth', 'gross_margins']],
            column_config={
                "industry": st.column_config.TextColumn("產業 (Industry)"),
                "Score": st.column_config.ProgressColumn("戰力分數 (Score)", min_value=0, max_value=100, format="%.1f"),
                "Buffett": st.column_config.TextColumn("巴菲特 (Buffett)"),
                "Quality": st.column_config.TextColumn("品質 (Quality)"),
                "rev_growth": st.column_config.NumberColumn("營收成長 (Rev Growth)", format="%.2f%%"),
                "eps_growth": st.column_config.NumberColumn("EPS成長 (EPS Growth)", format="%.2f%%"),
                "gross_margins": st.column_config.NumberColumn("毛利率 (Gross Margin)", format="%.2f%%"),
            },
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🎯 深度戰略分析")
        
        for idx, row in final_df.head(10).iterrows():
            code = row['代號']
            
            with st.container():
                industry_tag = f"<span class='tag tag-sector'>{row['industry']}</span>"
                buffett_tag = "<span class='tag tag-buffett'>Buffett Pick</span>" if row['Buffett'] else ""
                quality_tag = ""
                if row['Quality'] == 'Profitless': quality_tag = "<span class='tag tag-warn'>⚠️ 虛胖警告 (Profitless)</span>"
                elif row['Quality'] == 'Quality': quality_tag = "<span class='tag tag-quality'>💎 高品質 (Quality)</span>"
                
                st.markdown(f"""
                <div class='stock-card'>
                    <div class='card-header'>
                        <div>
                            <span class='header-title'>{row['名稱']} ({code})</span>
                            <span class='header-price'>${row['close_price']}</span>
                        </div>
                        <div>{industry_tag}{buffett_tag}{quality_tag}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 1.5, 1.5])
                
                with c1:
                    if idx in df_norm.index:
                        st.plotly_chart(plot_radar_chart_ui(row['名稱'], get_radar_data(df_norm.loc[idx])), use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                    <div class='metrics-grid'>
                        <div class='metric-item'><span class='m-label'>營收成長 (Rev Growth)</span><span class='m-val m-high'>{row.get('rev_growth', 0):.2f}%</span></div>
                        <div class='metric-item'><span class='m-label'>EPS 成長 (EPS Growth)</span><span class='m-val m-high'>{row.get('eps_growth', 0):.2f}%</span></div>
                        <div class='metric-item'><span class='m-label'>毛利率 (Gross Margin)</span><span class='m-val'>{row.get('gross_margins', 0):.2f}%</span></div>
                        <div class='metric-item'><span class='m-label'>PEG Ratio</span><span class='m-val'>{row.get('peg', 0):.2f}</span></div>
                        <div class='metric-item'><span class='m-label'>本益比 (PE)</span><span class='m-val'>{row.get('pe', 0):.2f}</span></div>
                        <div class='metric-item'><span class='m-label'>季線乖離 (MA Bias)</span><span class='m-val'>{row.get('priceToMA60', 0)*100:.1f}%</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    b1, b2 = st.columns(2)
                    if b1.button(f"✨ AI 分析", key=f"ai_{idx}"):
                        p_txt = AI_PROMPT.replace("[STOCK]", row['名稱']).replace("[SECTOR]", str(row['industry'])).replace("[PE]", str(row.get('pe'))).replace("[PEG]", str(row.get('peg'))).replace("[REV]", str(row.get('rev_growth'))).replace("[EPS_G]", str(row.get('eps_growth'))).replace("[GM]", str(row.get('gross_margins'))).replace("[ROE]", str(row.get('roe')))
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
