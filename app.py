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
import os

# --- 1. 介面與 CSS 設定 (視覺優化) ---
st.set_page_config(
    page_title="熵值決策選股平台 (Visual Master)", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全域黑底 */
    .stApp { background-color: #0e1117 !important; }
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, li { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
    
    /* 側邊欄優化 */
    div[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* 個股卡片風格 (依照您的截圖) */
    .stock-card { 
        background-color: #1f2937; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #374151; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .card-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        border-bottom: 1px solid #374151;
        padding-bottom: 10px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .strategy-tag {
        background-color: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: normal;
    }
    .buffett-tag {
        background-color: #FFD700;
        color: #000;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 5px;
    }
    
    /* 數據表格樣式 */
    .data-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        font-size: 0.9rem;
    }
    .data-item {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px dashed #4b5563;
        padding: 5px 0;
    }
    .val-pos { color: #4ade80; } /* 正向綠 */
    .val-neg { color: #f87171; } /* 負向紅 */
    
    /* 下載按鈕 */
    .stDownloadButton button { background-color: #374151 !important; border: 1px solid #4b5563 !important; color: white !important; }
    .stDownloadButton button:hover { border-color: #60a5fa !important; color: #60a5fa !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. Session State 初始化 ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None
# 這裡儲存歷史 K 線數據，確保趨勢圖有資料可畫
if 'history_storage' not in st.session_state: st.session_state['history_storage'] = {}

# --- 3. API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！請確認 Secrets 設定。")
    st.stop()

# --- 4. 字型下載 ---
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

# --- 5. 核心數據引擎 (Stable Schema + History) ---

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

def get_stock_info_single(symbol):
    """抓取基本面數據 (info)"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'pe': info.get('trailingPE'),
            'peg': info.get('pegRatio'),
            'pb': info.get('priceToBook'),
            'rev_growth': info.get('revenueGrowth'),
            'yield': info.get('dividendYield'),
            'roe': info.get('returnOnEquity'),
            'beta': info.get('beta'),
            'sector': info.get('sector', 'General')
        }
    except: return {}

def calculate_synthetic_peg(pe, growth_rate):
    """若 Yahoo 沒給 PEG，自己算"""
    if pe and growth_rate and growth_rate > 0:
        return pe / (growth_rate * 100)
    return None

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

@st.cache_data(ttl=300, show_spinner=False)
def batch_scan_stocks_v3(stock_list, tej_data=None):
    results = []
    history_storage = {} # 暫存 K 線數據供繪圖使用
    
    symbols = [s.split(' ')[0] for s in stock_list]
    
    # 1. 批量抓取 K 線 (保證有圖)
    try:
        # download 回傳的是 MultiIndex (Price, Symbol)
        batch_hist = yf.download(symbols, period="6mo", group_by='ticker', progress=False, threads=True)
    except:
        batch_hist = pd.DataFrame()

    # 2. 平行抓取基本面
    fundamentals_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_info_single, s): s for s in symbols}
        for future in concurrent.futures.as_completed(future_to_stock):
            s = future_to_stock[future]
            try: fundamentals_map[s] = future.result()
            except: fundamentals_map[s] = {}

    # 3. 整合與計算
    for stock_str in stock_list:
        symbol = stock_str.split(' ')[0]
        code = symbol.split('.')[0]
        name = stock_str.split(' ')[1] if len(stock_str.split(' ')) > 1 else code
        
        # 初始化
        price = np.nan; ma_bias = 0; volatility = 0.5
        pe = np.nan; pb = np.nan; dy = np.nan; rev_growth = np.nan; peg = np.nan; roe = np.nan
        chips = 0
        
        # A. 處理 K 線與技術指標
        try:
            if len(symbols) == 1: df_hist = batch_hist
            else: df_hist = batch_hist[symbol]
            
            # 清理並儲存
            if not df_hist.empty and 'Close' in df_hist.columns:
                df_clean = df_hist.dropna(subset=['Close'])
                if not df_clean.empty:
                    history_storage[code] = df_clean # 存起來畫圖
                    price = float(df_clean['Close'].iloc[-1])
                    # 波動率
                    volatility = df_clean['Close'].pct_change().std() * (252**0.5)
                    # MA60
                    ma60 = df_clean['Close'].rolling(60).mean().iloc[-1]
                    if not pd.isna(ma60): ma_bias = (price / ma60) - 1
        except: pass
        
        # B. 處理基本面
        fund = fundamentals_map.get(symbol, {})
        pe = fund.get('pe')
        pb = fund.get('pb')
        peg = fund.get('peg')
        roe = fund.get('roe')
        if fund.get('yield'): dy = fund.get('yield') * 100
        if fund.get('rev_growth'): rev_growth = fund.get('rev_growth') * 100
        
        # C. TEJ 覆蓋
        if tej_data and code in tej_data:
            t_row = tej_data[code]
            for k, v in t_row.items():
                if '法人' in k or 'Chips' in k: chips = float(v) if v != '-' else 0
        
        # D. 合成 PEG
        if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
            peg = calculate_synthetic_peg(pe, rev_growth/100)
            
        # E. 產業判讀
        industry = 'General'
        if code in ['2330', '2454', '2303', '3034', '3035', '2379', '2382', '3231']: industry = 'Semicon'
        elif code.startswith('28'): industry = 'Finance'
        elif code in ['1101', '1301', '2002', '2603']: industry = 'Cyclical'

        # 只要有代號就加入，確保不缺漏
        results.append({
            '代號': code, '名稱': name, 'full_symbol': symbol,
            'close_price': price,
            'pe': pe, 'pb': pb, 'yield': dy, 'roe': roe,
            'rev_growth': rev_growth, 'peg': peg, 'chips': chips,
            'volatility': volatility, 'priceToMA60': ma_bias,
            'industry': industry
        })
        
    # 強制建立 DataFrame (Robust Schema)
    expected_cols = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'roe', 
                     'rev_growth', 'peg', 'chips', 'volatility', 'priceToMA60', 
                     'industry', 'full_symbol']
    df_res = pd.DataFrame(results)
    for c in expected_cols:
        if c not in df_res.columns: df_res[c] = np.nan
        
    return df_res, history_storage

# --- 8. 熵值模型 ---
def get_sector_config(industry):
    config = {
        'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 1, 'cat': '風險'}, 
        'MA Bias': {'col': 'priceToMA60', 'dir': 'min', 'w': 1, 'cat': '技術'},
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
        config = get_sector_config(row.get('industry', 'General'))
        total_score = 0
        total_weight = 0
        
        for name, setting in config.items():
            all_vals = calc_df[setting['col']]
            rank = all_vals.rank(pct=True).get(idx, 0.5)
            if setting['dir'] == 'max': norm = rank
            else: norm = 1 - rank
            
            # 存入 df_norm
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
        ma = row.get('priceToMA60', 0)
        
        if final > 75 and rev > 20: plans.append("🚀 爆發成長")
        elif final > 60: plans.append("🟡 穩健持有")
        elif ma < -0.1: plans.append("🟢 超跌反彈")
        else: plans.append("⛔ 觀望")
            
    df['Score'] = scores
    df['Strategy'] = plans
    df['Buffett'] = buffett_tags
    return df.sort_values('Score', ascending=False), df_norm

# --- 9. 繪圖函數 (視覺回歸) ---
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

def plot_radar_chart_ui(radar_data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(radar_data.values()), theta=list(radar_data.keys()),
        fill='toself', line_color='#00e676', fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#374151'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=30, r=30), height=250,
        font=dict(color='#e6e6e6')
    )
    return fig

def plot_trend_chart_ui(history_df):
    if not isinstance(history_df, pd.DataFrame) or history_df.empty: 
        return None
    
    # 計算 MA60
    history_df['MA60'] = history_df['Close'].rolling(window=60).mean()
    
    fig = go.Figure()
    # 股價線
    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df['Close'], 
        name='Price', line=dict(color='#29b6f6', width=2)
    ))
    # 季線
    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df['MA60'], 
        name='MA60', line=dict(color='#fbbf24', width=1.5, dash='dash')
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, linecolor='#374151'),
        yaxis=dict(showgrid=True, gridcolor='#374151'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=0, r=0), height=200,
        showlegend=False,
        font=dict(color='#e6e6e6')
    )
    return fig

def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = [Paragraph(f"Analysis Report: {stock_data['名稱']}", getSampleStyleSheet()['Heading1'])]
    # Filter non-serializable
    safe = {k:v for k,v in stock_data.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    for k, v in safe.items():
        story.append(Paragraph(f"{k}: {v}", getSampleStyleSheet()['Normal']))
    try: doc.build(story)
    except: pass
    buffer.seek(0)
    return buffer

def call_ai(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers=headers, json=data)
        return r.json()['candidates'][0]['content']['parts'][0]['text']
    except: return "AI 分析連線失敗，請稍後再試。"

AI_PROMPT = """
請扮演華爾街基金經理人，分析 [STOCK] ([SECTOR])。
數據：PE=[PE], PEG=[PEG], 營收成長=[REV]%, ROE=[ROE]%, 波動率=[VOL]%.
重點：
1. **成長性**：是否具備爆發潛力？(PEG < 1.5 ?)
2. **安全性**：是否符合巴菲特護城河 (高ROE, 低波動)？
3. **結論**：給出操作建議 (買進/持有/賣出)。
"""

# --- 10. 主程式 (側邊欄與主畫面) ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    
    # 1. 資料源
    st.markdown("### 1️⃣ 資料源與匯入")
    with st.expander("📂 匯入 TEJ (選填)"):
        uploaded = st.file_uploader("上傳 CSV/Excel", type=['csv','xlsx'])
        if uploaded: 
            st.session_state['tej_data'] = process_tej_upload(uploaded)
            st.success(f"已載入 TEJ 數據")

    # 2. 策略
    st.markdown("### 2️⃣ 策略設定")
    use_buffett = st.checkbox("🎩 啟用巴菲特選股邏輯", value=False)
    
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
        with st.spinner("正在執行雙軌掃描 (Yahoo K線 + 財報挖掘)..."):
            # 呼叫 v3 掃描
            raw, hist_store = batch_scan_stocks_v3(target_stocks, st.session_state['tej_data'])
            st.session_state['raw_data'] = raw
            st.session_state['history_storage'] = hist_store
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 25.0")
    st.caption("Visual Dashboard + Robust Data + Buffett Logic")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    hist_storage = st.session_state.get('history_storage', {})
    
    if df.empty:
        st.error("❌ 查無數據，請檢查代號或網路。")
    else:
        final_df, df_norm = calculate_score(df, use_buffett)
        
        # 1. 總表
        st.subheader("🏆 潛力標的排行")
        st.dataframe(
            final_df[['代號', '名稱', 'industry', 'Score', 'Buffett', 'Strategy', 'pe', 'rev_growth', 'peg', 'yield']],
            column_config={
                "industry": st.column_config.TextColumn("產業"),
                "Score": st.column_config.ProgressColumn("戰力分數", min_value=0, max_value=100, format="%.1f"),
                "Buffett": st.column_config.TextColumn("巴菲特"),
                "rev_growth": st.column_config.NumberColumn("營收成長", format="%.2f%%"),
                "peg": st.column_config.NumberColumn("PEG"),
                "yield": st.column_config.NumberColumn("殖利率", format="%.2f%%"),
            },
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🎯 深度戰略分析 (Dashboard View)")
        
        # 2. 個股卡片 (Visual Loop)
        for idx, row in final_df.head(5).iterrows(): # 顯示前5名
            code = row['代號']
            
            with st.container():
                # 標籤區
                ind_html = f"<span class='sector-tag'>{row['industry']}</span>"
                buf_html = "<span class='buffett-tag'>Buffett Pick</span>" if row['Buffett'] else ""
                
                # 卡片頭部
                st.markdown(f"""
                <div class='stock-card'>
                    <div class='card-header'>
                        <span>{row['名稱']} ({code})</span>
                        <div>{ind_html}{buf_html}<span class='strategy-tag'>{row['Strategy']}</span></div>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 1.2, 1.5])
                
                # 左：雷達圖
                if idx in df_norm.index:
                    radar_data = get_radar_data(df_norm.loc[idx])
                    with c1:
                        st.markdown("**五力分析**")
                        st.plotly_chart(plot_radar_chart_ui(radar_data), use_container_width=True)
                
                # 中：數據網格
                with c2:
                    st.markdown("**關鍵指標**")
                    st.markdown(f"""
                    <div class='data-grid'>
                        <div class='data-item'><span>本益比 (PE)</span> <span class='val-pos'>{row.get('pe', 'N/A')}</span></div>
                        <div class='data-item'><span>殖利率</span> <span class='val-pos'>{row.get('yield', 0):.2f}%</span></div>
                        <div class='data-item'><span>營收成長</span> <span class='val-pos'>{row.get('rev_growth', 0):.2f}%</span></div>
                        <div class='data-item'><span>PEG</span> <span>{row.get('peg', 'N/A')}</span></div>
                        <div class='data-item'><span>波動率</span> <span class='val-neg'>{row.get('volatility', 0)*100:.1f}%</span></div>
                        <div class='data-item'><span>季線乖離</span> <span>{row.get('priceToMA60', 0)*100:.1f}%</span></div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)
                    
                    # 按鈕
                    if st.button(f"✨ AI 解讀", key=f"ai_{idx}"):
                        p_txt = AI_PROMPT.replace("[STOCK]", row['名稱']).replace("[SECTOR]", row['industry']).replace("[PE]", str(row.get('pe'))).replace("[PEG]", str(row.get('peg'))).replace("[REV]", str(row.get('rev_growth'))).replace("[ROE]", str(row.get('roe'))).replace("[VOL]", str(round(row.get('volatility',0)*100,1)))
                        an = call_ai(p_txt)
                        st.info(an)
                        
                    pdf = create_pdf(row.to_dict())
                    st.download_button("📥 下載報告", pdf, f"{code}.pdf", key=f"dl_{idx}")

                # 右：趨勢圖
                with c3:
                    st.markdown("**趨勢診斷 (MA60)**")
                    if code in hist_storage and not hist_storage[code].empty:
                        st.plotly_chart(plot_trend_chart_ui(hist_storage[code]), use_container_width=True)
                    else:
                        st.warning("暫無 K 線數據")

                st.markdown("</div>", unsafe_allow_html=True) # End Card

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動全自動掃描」開始。")
