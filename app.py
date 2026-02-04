import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import io
import time
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import requests
import os

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="熵值決策選股平台 (Stable)", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, li { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
    .metric-card { background-color: #1f2937; padding: 15px; border-radius: 8px; border: 1px solid #374151; text-align: center; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #00e676; }
    .metric-label { font-size: 0.9rem; color: #9ca3af; }
    div[data-testid="stExpander"] { background-color: #1f2937 !important; border: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None

# --- 4. 字型下載 (PDF用) ---
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
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
            return True
    except: return False
    return False

font_ready = register_chinese_font()

# --- 5. 核心數據引擎 (Yahoo Finance Deep Fetch) ---
def get_stock_fundamentals(symbol):
    """
    從 Yahoo Finance 抓取完整的財務數據
    """
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): 
            symbol += '.TW'
        
        ticker = yf.Ticker(symbol)
        # 使用 fast_info 獲取價格 (較快)，info 獲取財報 (較慢但詳細)
        info = ticker.info 
        
        # 提取關鍵數據，若無則回傳 None
        data = {
            'close_price': info.get('currentPrice') or info.get('previousClose'),
            'pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg': info.get('pegRatio'),
            'pb': info.get('priceToBook'),
            'roe': info.get('returnOnEquity'),
            'rev_growth': info.get('revenueGrowth'), # 營收成長 YoY
            'yield': info.get('dividendYield'), # 殖利率
            'sector': info.get('sector', 'Unknown'),
            'beta': info.get('beta')
        }
        return data
    except Exception as e:
        return None

def calculate_synthetic_peg(pe, growth_rate):
    """計算合成 PEG"""
    if pe and growth_rate and growth_rate > 0:
        return pe / (growth_rate * 100)
    return None

def process_tej_upload(uploaded_file):
    """處理 TEJ 上傳檔案"""
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

# --- 6. 批量掃描邏輯 ---
@st.cache_data(ttl=600, show_spinner=False)
def batch_scan_stocks(stock_list, tej_data=None):
    results = []
    
    # 定義標準欄位，防止空表導致 KeyError
    columns = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 
               'rev_growth', 'roe', 'peg', 'chips', 'industry']

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_fundamentals, s.split(' ')[0]): s for s in stock_list}
        
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            try:
                code = stock_str.split(' ')[0].split('.')[0]
                name = stock_str.split(' ')[1] if len(stock_str.split(' ')) > 1 else code
                
                y_data = future.result()
                
                # 初始化變數
                price = np.nan; pe = np.nan; pb = np.nan; dy = np.nan
                rev_growth = np.nan; roe = np.nan; peg = np.nan
                chips = 0
                
                # 1. 填入 Yahoo 數據
                if y_data:
                    price = y_data.get('close_price')
                    pe = y_data.get('pe')
                    pb = y_data.get('pb')
                    if y_data.get('yield'): dy = y_data.get('yield') * 100
                    if y_data.get('rev_growth'): rev_growth = y_data.get('rev_growth') * 100
                    if y_data.get('roe'): roe = y_data.get('roe') * 100
                    peg = y_data.get('peg')
                
                # 2. 若 Yahoo 有缺，嘗試用 TEJ 補
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    for k, v in t_row.items():
                        if '本益比' in k or 'PE' in k: pe = float(v) if v != '-' else pe
                        if '淨值比' in k or 'PB' in k: pb = float(v) if v != '-' else pb
                        if '殖利率' in k or 'Yield' in k: dy = float(v) if v != '-' else dy
                        if '營收成長' in k or 'Growth' in k: rev_growth = float(v) if v != '-' else rev_growth
                        if '法人' in k or 'Chips' in k: chips = float(v) if v != '-' else chips

                # 3. 計算合成 PEG
                if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
                    peg = calculate_synthetic_peg(pe, rev_growth/100)

                # 只有當至少有股價時才加入列表
                if not pd.isna(price):
                    results.append({
                        '代號': code, '名稱': name, 
                        'close_price': price,
                        'pe': pe, 'pb': pb, 'yield': dy,
                        'rev_growth': rev_growth, 'roe': roe, 'peg': peg,
                        'chips': chips,
                        'industry': 'Semicon' if code in ['2330', '2454', '2303'] else ('Finance' if code.startswith('28') else 'General')
                    })
                
            except Exception as e:
                continue # 單檔失敗不影響整體
    
    # 【關鍵修復】即使 results 為空，也要回傳有欄位的 DataFrame
    if not results:
        return pd.DataFrame(columns=columns)
                
    return pd.DataFrame(results)

# --- 7. 熵值模型與評分 ---
def get_entropy_config(industry):
    config = {
        'P/E': {'col': 'pe', 'dir': 'min', 'w': 1},
        'P/B': {'col': 'pb', 'dir': 'min', 'w': 1},
        'Yield': {'col': 'yield', 'dir': 'max', 'w': 1},
    }
    if industry == 'Semicon':
        config['Rev Growth'] = {'col': 'rev_growth', 'dir': 'max', 'w': 2}
        config['PEG'] = {'col': 'peg', 'dir': 'min', 'w': 1.5}
    elif industry == 'Finance':
        config['Yield']['w'] = 2 
        config['P/B']['w'] = 1.5
    else:
        config['P/E']['w'] = 1.5
    return config

def calculate_score(df):
    if df.empty: return df
    
    scores = []
    action_plans = []
    
    # 確保所需欄位存在 (防止 KeyError)
    for col in ['rev_growth', 'peg', 'pe', 'pb', 'yield']:
        if col not in df.columns:
            df[col] = np.nan

    for idx, row in df.iterrows():
        config = get_entropy_config(row.get('industry', 'General'))
        total_score = 0
        total_weight = 0
        
        for name, setting in config.items():
            val = row.get(setting['col'])
            if pd.isna(val) or val == 0:
                score = 50
            else:
                all_vals = df[setting['col']].dropna()
                if all_vals.empty:
                    score = 50
                else:
                    rank_pct = all_vals.rank(pct=True).get(idx, 0.5) 
                    if setting['dir'] == 'max':
                        score = rank_pct * 100
                    else:
                        score = (1 - rank_pct) * 100
            
            total_score += score * setting['w']
            total_weight += setting['w']
            
        final_score = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final_score, 1))
        
        rev_g = row.get('rev_growth', 0)
        peg = row.get('peg', 100)
        if pd.isna(peg): peg = 100
        if pd.isna(rev_g): rev_g = 0
        
        if final_score > 70 and rev_g > 20 and peg < 1.2:
            action_plans.append("🚀 爆發成長 (Buy)")
        elif final_score > 60:
            action_plans.append("🟡 穩健持有 (Hold)")
        else:
            action_plans.append("⛔ 觀望 (Wait)")
            
    df['Entropy Score'] = scores
    df['Strategy'] = action_plans
    return df.sort_values('Entropy Score', ascending=False)

# --- 8. 主程式 ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    st.markdown("---")
    
    with st.expander("📂 匯入 TEJ 籌碼/財報 (選填)"):
        st.caption("Yahoo 無法提供台股每日籌碼，建議匯入 TEJ 檔案以獲得完整分析。")
        uploaded_file = st.file_uploader("上傳 Excel/CSV", type=['csv', 'xlsx'])
        if uploaded_file:
            tej_data = process_tej_upload(uploaded_file)
            if tej_data: st.session_state['tej_data'] = tej_data
    
    strategy = st.selectbox("選股策略:", ["台灣50 (大型股)", "AI 供應鏈 (成長)", "高股息 (價值)"])
    
    if strategy == "台灣50 (大型股)":
        target_stocks = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海", "2308.TW 台達電", "2881.TW 富邦金"]
    elif strategy == "AI 供應鏈 (成長)":
        target_stocks = ["2330.TW 台積電", "2382.TW 廣達", "3231.TW 緯創", "3017.TW 奇鋐", "3661.TW 世芯-KY"]
    elif strategy == "高股息 (價值)":
        target_stocks = ["2301.TW 光寶科", "2454.TW 聯發科", "3034.TW 聯詠", "2886.TW 兆豐金", "1101.TW 台泥"]
        
    if st.button("🚀 啟動掃描 (使用 Yahoo Deep Fetch)", type="primary"):
        st.session_state['scan_finished'] = False
        with st.spinner("正在深入挖掘 Yahoo 財報數據 (速度較慢請稍候)..."):
            raw = batch_scan_stocks(target_stocks, st.session_state.get('tej_data'))
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

# 主畫面
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 19.1")
    st.caption("Yahoo Finance API + Dynamic Industry Weighting")

if st.session_state['scan_finished']:
    df = st.session_state['raw_data']
    
    if df is None or df.empty:
        st.error("❌ 掃描失敗：無法獲取任何股票數據。可能是 Yahoo API 暫時阻擋或網路問題。")
    else:
        # 計算分數
        final_df = calculate_score(df)
        
        st.subheader("🏆 潛力標的排行 (Entropy Ranking)")
        
        display_df = final_df.copy()
        
        # 顯示表格 (確保所有欄位都存在)
        st.dataframe(
            display_df[['代號', '名稱', 'close_price', 'Entropy Score', 'Strategy', 'pe', 'rev_growth', 'peg', 'yield', 'chips']],
            column_config={
                "Entropy Score": st.column_config.ProgressColumn("綜合戰力", min_value=0, max_value=100, format="%.1f"),
                "close_price": st.column_config.NumberColumn("股價", format="$%.1f"),
                "pe": st.column_config.NumberColumn("本益比", format="%.1f"),
                "rev_growth": st.column_config.NumberColumn("營收成長(%)", format="%.2f%%"),
                "peg": st.column_config.NumberColumn("PEG", format="%.2f"),
                "yield": st.column_config.NumberColumn("殖利率(%)", format="%.2f%%"),
                "chips": st.column_config.NumberColumn("法人買賣超", help="需匯入 TEJ 才有數據"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.info("💡 提示：若數據為空，代表 Yahoo 資料庫暫無該股資料 (通常發生在剛上市或冷門股)。")

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動掃描」開始分析。")
