import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import requests
import io
import time
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="熵值決策選股平台 (Chart & UI Fix)", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 強制修復 (選單可見性 + 專業配色) ---
st.markdown("""
<style>
    /* 1. 全域背景：深邃黑 */
    .stApp { background-color: #0e1117 !important; }
    
    /* 2. 主文字顏色 */
    body, h1, h2, h3, h4, h5, h6, p, label { 
        color: #e6e6e6 !important; 
        font-family: 'Roboto', sans-serif; 
    }
    
    /* 3. 側邊欄 */
    [data-testid="stSidebar"] { 
        background-color: #161b22 !important; 
        border-right: 1px solid #30363d;
    }
    
    /* 4. 【關鍵修復】下拉選單 (Multiselect) 強制黑底白字 */
    div[role="listbox"] ul {
        background-color: #262730 !important;
    }
    li[role="option"] {
        color: white !important;
        background-color: #262730 !important;
    }
    li[role="option"]:hover {
        background-color: #238636 !important; /* 綠色高亮 */
    }
    /* 選中項目的標籤 */
    span[data-baseweb="tag"] {
        background-color: #1f2937 !important;
        color: #e6e6e6 !important;
    }
    
    /* 5. 輸入框 */
    input { 
        background-color: #0d1117 !important; 
        color: white !important; 
        border: 1px solid #30363d !important; 
    }
    
    /* 6. 專業個股卡片 */
    .stock-card { 
        background-color: #1f2937; 
        padding: 20px; 
        border-radius: 8px; 
        border: 1px solid #374151; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    }
    .card-header-text {
        font-size: 1.4rem; font-weight: 700; color: #ffffff !important;
    }
    
    /* 7. 標籤 */
    .buffett-tag { 
        background-color: #FFD700; color: #000 !important; 
        padding: 4px 8px; border-radius: 4px; 
        font-weight: 800; font-size: 0.75rem; 
    }
    .sector-tag {
        background-color: #238636; color: #fff !important;
        padding: 4px 8px; border-radius: 4px;
        font-size: 0.75rem;
    }
    
    /* 8. 按鈕 */
    .stButton button { 
        background-color: #238636; color: white; border: none; font-weight: bold;
    }
    .stButton button:hover { background-color: #2ea043; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None
# 【關鍵】獨立儲存 K 線數據，避免在 DataFrame 傳遞中遺失
if 'history_storage' not in st.session_state: st.session_state['history_storage'] = {}

# --- 4. API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！")
    st.stop()

# --- 5. 字型 ---
@st.cache_resource
def register_chinese_font():
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
    return False

font_ready = register_chinese_font()

# --- 6. 核心數據引擎 ---
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

def get_stock_data_full(symbol):
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): symbol += '.TW'
        ticker = yf.Ticker(symbol)
        info = ticker.info 
        
        data = {
            'close_price': info.get('currentPrice') or info.get('previousClose'),
            'pe': info.get('trailingPE'),
            'peg': info.get('pegRatio'),
            'pb': info.get('priceToBook'),
            'rev_growth': info.get('revenueGrowth'),
            'yield': info.get('dividendYield'),
            'roe': info.get('returnOnEquity'),
            'beta': info.get('beta'),
            'sector': info.get('sector', 'General')
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

def process_tej_upload(uploaded_file):
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
        else: df = pd.read_excel(uploaded_file)
        df.columns = [str(c).strip() for c in df.columns]
        code_col = next((c for c in df.columns if '代號' in c or 'Code' in c), None)
        if not code_col: return None
        tej_map = {}
        for _, row in df.iterrows():
            raw_code = str(row[code_col]).split('.')[0].strip()
            tej_map[raw_code] = row.to_dict()
        return tej_map
    except: return None

# --- 7. 批量掃描 (分離儲存 K 線) ---
@st.cache_data(ttl=300, show_spinner=False)
def batch_scan_stocks_v4(stock_list, tej_data=None):
    results = []
    # 這裡只回傳「乾淨」的 DataFrame，不含 K 線物件
    # K 線物件會另外回傳一個 dict
    history_storage = {} 
    
    # 1. 批量抓 K 線
    try:
        symbols = [s.split(' ')[0] + ('.TW' if not s.endswith('.TW') else '') for s in stock_list]
        hist_data = yf.download(symbols, period="6mo", group_by='ticker', progress=False, threads=True)
    except: hist_data = pd.DataFrame()

    # 2. 抓基本面
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_data_full, s.split(' ')[0]): s for s in stock_list}
        
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            try:
                code = stock_str.split(' ')[0].split('.')[0]
                name = stock_str.split(' ')[1] if len(stock_str.split(' ')) > 1 else code
                y_data = future.result()
                
                # 初始化
                price = np.nan; pe = np.nan; pb = np.nan; dy = np.nan
                rev_growth = np.nan; peg = np.nan; roe = np.nan; volatility = 0.5
                chips = 0
                ma_bias = 0

                # 處理 K 線與畫圖數據
                try:
                    s_sym = f"{code}.TW"
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        if s_sym in hist_data: df_hist = hist_data[s_sym]
                        else: df_hist = pd.DataFrame()
                    else:
                        df_hist = hist_data if len(symbols) == 1 else pd.DataFrame()

                    if not df_hist.empty and 'Close' in df_hist.columns:
                        closes = df_hist['Close'].dropna()
                        if len(closes) > 10:
                            price = float(closes.iloc[-1])
                            volatility = closes.pct_change().std() * (252**0.5)
                            ma60 = closes.rolling(60).mean().iloc[-1]
                            if not pd.isna(ma60): ma_bias = (price / ma60) - 1
                            
                            # 【關鍵】將 K 線存入字典，而非 DataFrame
                            history_storage[code] = df_hist 
                except: pass

                if y_data:
                    if pd.isna(price): price = y_data.get('close_price')
                    pe = y_data.get('pe')
                    pb = y_data.get('pb')
                    roe = y_data.get('roe')
                    raw_dy = y_data.get('yield')
                    if raw_dy: dy = raw_dy * 100 
                    raw_rev = y_data.get('rev_growth')
                    if raw_rev: rev_growth = raw_rev * 100
                    peg = y_data.get('peg')
                
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    for k, v in t_row.items():
                        if '法人' in k: chips = float(v) if v != '-' else 0

                if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
                    peg = calculate_synthetic_peg(pe, rev_growth/100)

                industry = 'General'
                if code in ['2330', '2454', '2303', '3034', '3035', '2379', '2382', '3231']: industry = 'Semicon'
                elif code.startswith('28'): industry = 'Finance'
                elif code in ['1101', '1301', '2002', '2603', '1802']: industry = 'Cyclical'

                if not pd.isna(price):
                    results.append({
                        '代號': code, '名稱': name, 'close_price': price,
                        'pe': pe, 'pb': pb, 'yield': dy, 'roe': roe,
                        'rev_growth': rev_growth, 'peg': peg, 'chips': chips,
                        'volatility': volatility, 'priceToMA60': ma_bias,
                        'industry': industry
                    })
            except: continue
    
    df = pd.DataFrame(results)
    # 強制 Schema
    cols = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'roe', 'rev_growth', 'peg', 'chips', 'volatility', 'priceToMA60', 'industry']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
        
    return df, history_storage

# --- 8. 評分邏輯 ---
def get_sector_config(industry):
    config = {
        'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 1, 'cat': '風險'}, 
    }
    if industry == 'Semicon': 
        config.update({
            'PEG': {'col': 'peg', 'dir': 'min', 'w': 2.0, 'cat': '成長'},
            'Rev Growth': {'col': 'rev_growth', 'dir': 'max', 'w': 1.5, 'cat': '成長'},
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
    df_norm = df.copy()
    scores = []
    plans = []
    buffett_tags = []
    
    fill_map = {'pe': 50, 'pb': 5, 'yield': 0, 'rev_growth': 0, 'peg': 5, 'volatility': 0.5, 'roe': 0, 'priceToMA60': 0}
    calc_df = df.fillna(fill_map)

    for idx, row in calc_df.iterrows():
        config = get_sector_config(row['industry'])
        total_score = 0
        total_weight = 0
        
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
        buffett_tags.append(is_buffett)
        if use_buffett and is_buffett:
            final += 15
            if final > 100: final = 100
            
        scores.append(round(final, 1))
        
        rev = row.get('rev_growth', 0)
        peg = row.get('peg', 100)
        if final > 75 and rev > 20: plans.append("🚀 爆發成長")
        elif final > 60: plans.append("🟡 穩健持有")
        else: plans.append("⛔ 觀望")
            
    df['Score'] = scores
    df['Strategy'] = plans
    df['Buffett'] = buffett_tags
    return df.sort_values('Score', ascending=False), df_norm

# --- 9. 繪圖函數 (高對比 + 黑色文字) ---
def get_radar_data(df_norm_row):
    cats = {'價值': 0, '成長': 0, '動能': 0, '風險': 0, '財報': 0}
    counts = {'價值': 0, '成長': 0, '動能': 0, '風險': 0, '財報': 0}
    for col in df_norm_row.index:
        if str(col).endswith('_n'):
            cat = str(col).split('_')[0]
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
        fill='toself', name=title, 
        line_color='#00e676', # 螢光綠
        fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#4b5563'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=30, r=30), height=250,
        font=dict(color='#e6e6e6')
    )
    return fig

def plot_trend_chart_ui(title, history_df):
    if history_df is None or history_df.empty: return None
    history_df['MA60'] = history_df['Close'].rolling(window=60).mean()
    
    fig = go.Figure()
    # 亮藍色股價
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='Price', line=dict(color='#29b6f6', width=2)))
    # 金黃色季線
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['MA60'], name='MA60', line=dict(color='#ffca28', width=1.5, dash='dash')))
    
    fig.update_layout(
        title=dict(text=f"{title} 趨勢", font=dict(color='white')),
        xaxis=dict(showgrid=False, linecolor='#4b5563', tickfont=dict(color='#9ca3af')),
        yaxis=dict(showgrid=True, gridcolor='#374151', tickfont=dict(color='#9ca3af')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=20, l=0, r=0), height=250,
        showlegend=False
    )
    return fig

def call_ai(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers=headers, json=data)
        return r.json()['candidates'][0]['content']['parts'][0]['text']
    except: return "AI 分析連線失敗。"

def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = [Paragraph(f"Analysis: {stock_data['名稱']}", getSampleStyleSheet()['Heading1'])]
    safe = {k:v for k,v in stock_data.items() if not isinstance(v, pd.DataFrame)}
    for k, v in safe.items():
        story.append(Paragraph(f"{k}: {v}", getSampleStyleSheet()['Normal']))
    try: doc.build(story)
    except: pass
    buffer.seek(0)
    return buffer

AI_PROMPT = """
請扮演華爾街基金經理人，分析 [STOCK] ([SECTOR])。
重點：
1. **成長性**：營收成長=[REV]%, PEG=[PEG]。
2. **安全性**：是否符合巴菲特護城河 (高ROE, 低波動)？
3. **結論**：操作建議。
"""

# --- 10. 主程式 ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    
    st.markdown("### 1️⃣ 數據源與匯入")
    with st.expander("📂 匯入 TEJ (選填)"):
        uploaded = st.file_uploader("上傳 CSV/Excel", type=['csv','xlsx'])
        if uploaded: 
            st.session_state['tej_data'] = process_tej_upload(uploaded)
            st.success(f"已載入 TEJ 數據")

    st.markdown("### 2️⃣ 策略設定")
    use_buffett = st.checkbox("🎩 啟用巴菲特選股", value=False)
    
    scan_mode = st.radio("模式選擇", ["🔥 熱門策略", "🏭 產業掃描", "⌨️ 自訂輸入"])
    
    target_stocks = []
    if scan_mode == "⌨️ 自訂輸入":
        default = ["2330.TW 台積電", "2317.TW 鴻海", "2454.TW 聯發科", "2881.TW 富邦金"]
        options = sorted(list(stock_map.values())) if stock_map else default
        selected = st.multiselect("選擇股票", options, default=[s for s in default if s in options])
        manual = st.text_input("手動輸入代號", "")
        target_stocks = selected
        if manual: target_stocks.append(f"{manual}.TW")
        
    elif scan_mode == "🏭 產業掃描":
        if not industry_map: st.error("⚠️ 無法載入產業列表，請改用自訂輸入。")
        else:
            ind_list = sorted(list(industry_map.keys()))
            ind = st.selectbox("選擇產業", ind_list)
            if ind: target_stocks = [stock_map[c] for c in industry_map[ind] if c in stock_map]
    else:
        strat = st.selectbox("策略集", ["台灣50", "AI供應鏈", "金融股"])
        if strat == "台灣50": target_stocks = ["2330.TW", "2454.TW", "2317.TW", "2308.TW", "2881.TW"]
        elif strat == "AI供應鏈": target_stocks = ["2330.TW", "2382.TW", "3231.TW", "3017.TW", "3661.TW"]
        else: target_stocks = ["2881.TW", "2882.TW", "2886.TW", "2891.TW"]

    if st.button("🚀 啟動全自動掃描", type="primary"):
        st.session_state['scan_finished'] = False
        with st.spinner("正在挖掘 Yahoo 數據 (含股價、財報、趨勢)..."):
            # V4 掃描：分離 DataFrame 與 K 線字典
            raw, hist_store = batch_scan_stocks_v4(target_stocks, st.session_state['tej_data'])
            raw = sanitize_data(raw)
            st.session_state['raw_data'] = raw
            st.session_state['history_storage'] = hist_store
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 27.0")
    st.caption("Visual Fix + Data Independent Storage")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {}) # 讀取 K 線字典
    
    if df.empty:
        st.error("❌ 查無數據，請檢查代號或網路。")
    else:
        final_df, df_norm = calculate_score(df, use_buffett)
        
        st.subheader("🏆 潛力標的排行")
        st.dataframe(
            final_df[['代號', '名稱', 'industry', 'Score', 'Buffett', 'Strategy', 'pe', 'rev_growth', 'peg', 'yield']],
            column_config={
                "industry": st.column_config.TextColumn("產業"),
                "Score": st.column_config.ProgressColumn("戰力分數", min_value=0, max_value=100, format="%.1f"),
                "Buffett": st.column_config.TextColumn("巴菲特"),
                "rev_growth": st.column_config.NumberColumn("營收成長", format="%.2f%%"),
                "peg": st.column_config.NumberColumn("PEG", format="%.2f"),
                "yield": st.column_config.NumberColumn("殖利率", format="%.2f%%"),
            },
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🎯 深度戰略分析")
        
        for idx, row in final_df.head(10).iterrows():
            code = row['代號']
            with st.container():
                industry_tag = f"<span class='sector-tag'>{row['industry']}</span>"
                buffett_tag = "<span class='buffett-tag'>Buffett Pick</span>" if row['Buffett'] else ""
                
                st.markdown(f"<div class='stock-card'><h3>{row['名稱']} ({code}) {industry_tag}{buffett_tag}</h3>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 1.5, 1.5])
                
                if idx in df_norm.index:
                    radar_data = get_radar_data(df_norm.loc[idx])
                    with c1:
                        st.plotly_chart(plot_radar_chart_ui(row['名稱'], radar_data), use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                    #### 關鍵數據
                    - **成長**: 營收成長 <span style='color:#4ade80'>{row.get('rev_growth', 0):.2f}%</span> | PEG {row.get('peg', 0):.2f}
                    - **價值**: 本益比 {row.get('pe', 0):.2f} | 殖利率 <span style='color:#4ade80'>{row.get('yield', 0):.2f}%</span>
                    - **風險**: 波動率 {row.get('volatility', 0)*100:.1f}%
                    """, unsafe_allow_html=True)
                    
                    b1, b2 = st.columns(2)
                    if b1.button(f"✨ AI 分析", key=f"ai_{idx}"):
                        p_txt = AI_PROMPT.replace("[STOCK]", row['名稱']).replace("[SECTOR]", str(row['industry'])).replace("[PE]", str(row.get('pe'))).replace("[PEG]", str(row.get('peg'))).replace("[REV]", str(row.get('rev_growth'))).replace("[ROE]", str(row.get('roe')))
                        an = call_ai(p_txt)
                        st.info(an)
                    
                    pdf = create_pdf(row.to_dict())
                    b2.download_button("📥 下載報告", pdf, f"{code}.pdf", key=f"dl_{idx}")

                with c3:
                    # 【關鍵】從字典中讀取 K 線畫圖
                    if code in hist_storage and not hist_storage[code].empty:
                        st.plotly_chart(plot_trend_chart_ui(row['名稱'], hist_storage[code]), use_container_width=True)
                    else:
                        st.warning("無歷史股價數據")

                st.markdown("</div>", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動全自動掃描」開始。")
