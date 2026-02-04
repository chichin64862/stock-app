import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import twstock
import concurrent.futures
import requests
import json
import time
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="熵值決策選股及AI深度分析平台 (Final Fix)", 
    page_icon="🔥", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, li { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
    div[role="menu"] div, div[role="menu"] span, div[role="menu"] label { color: #31333F !important; font-weight: 500 !important; }
    div[data-baseweb="select"] > div { background-color: #262730 !important; border-color: #4b4b4b !important; color: white !important; }
    .stDownloadButton button { background-color: #1f2937 !important; border: 1px solid #238636 !important; width: 100%; }
    .stDownloadButton button:hover { border-color: #58a6ff !important; color: #58a6ff !important; }
    .stock-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 15px; }
    .ai-header { color: #58a6ff !important; font-weight: bold; font-size: 1.3rem; margin-bottom: 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    div[data-testid="stExpander"] { background-color: #1f2937 !important; border: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'df_norm' not in st.session_state: st.session_state['df_norm'] = None
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None

# --- 4. API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！請確認您已在 Streamlit Cloud > Settings > Secrets 中設定 `GEMINI_API_KEY`。")
    st.stop()

# --- 5. 字型下載 ---
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

# --- 6. 核心數據引擎 (分離式架構) ---

@st.cache_data
def get_tw_stock_info():
    """取得台股代號列表"""
    codes = twstock.codes
    stock_dict = {} 
    industry_dict = {} 
    for code, info in codes.items():
        if info.type == '股票':
            if info.market == '上市': suffix = '.TW'
            elif info.market == '上櫃': suffix = '.TWO'
            else: continue
            full_code = f"{code}{suffix}"
            name = info.name
            industry = info.group
            stock_dict[full_code] = f"{full_code} {name}"
            if industry not in industry_dict: industry_dict[industry] = []
            industry_dict[industry].append(full_code)
    return stock_dict, industry_dict

stock_map, industry_map = get_tw_stock_info()

def get_fundamental_info(symbol):
    """第二層：單獨抓取基本面 (Info) - 容易失敗，需獨立處理"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'pe': info.get('trailingPE'),
            'pb': info.get('priceToBook'),
            'peg': info.get('pegRatio'),
            'rev_growth': info.get('revenueGrowth'),
            'yield': info.get('dividendYield')
        }
    except:
        return {}

def calculate_synthetic_peg(pe, growth_rate):
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

@st.cache_data(ttl=600, show_spinner=False)
def batch_scan_stocks_v2(stock_list, tej_data=None):
    """
    V2 掃描邏輯：
    1. 先用 yf.download 批量抓 K 線 (這部分最穩，保證有股價)。
    2. 再用 ThreadPool 補抓 info (財報)。
    3. 最後合併，確保至少有價格和技術指標。
    """
    results = []
    symbols = [s.split(' ')[0] for s in stock_list]
    
    # 1. 批量抓取價格與技術面 (穩如泰山)
    try:
        # 下載 6 個月數據以計算 MA60 和波動率
        history_data = yf.download(symbols, period="6mo", group_by='ticker', progress=False, threads=True)
    except:
        history_data = pd.DataFrame()

    # 2. 平行抓取基本面 (盡力而為)
    fundamentals_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_fundamental_info, s): s for s in symbols}
        for future in concurrent.futures.as_completed(future_to_stock):
            s = future_to_stock[future]
            try:
                fundamentals_map[s] = future.result()
            except:
                fundamentals_map[s] = {}

    # 3. 整合數據
    for stock_str in stock_list:
        symbol = stock_str.split(' ')[0]
        code = symbol.split('.')[0]
        name = stock_str.split(' ')[1] if len(stock_str.split(' ')) > 1 else code
        
        # 初始化
        price = np.nan; ma_bias = 0; volatility = 0
        pe = np.nan; pb = np.nan; dy = np.nan; rev_growth = np.nan; peg = np.nan; chips = 0
        
        # A. 填入技術面
        try:
            if len(symbols) == 1: df = history_data
            else: df = history_data[symbol]
            
            if not df.empty and 'Close' in df.columns:
                # 移除空值行
                df = df.dropna(subset=['Close'])
                if not df.empty:
                    price = float(df['Close'].iloc[-1])
                    # 計算 MA60 乖離
                    ma60 = df['Close'].rolling(60).mean().iloc[-1]
                    if not pd.isna(ma60): ma_bias = (price / ma60) - 1
                    # 計算波動率
                    volatility = df['Close'].pct_change().std() * (252 ** 0.5)
        except: pass
        
        # B. 填入基本面
        fund = fundamentals_map.get(symbol, {})
        pe = fund.get('pe')
        pb = fund.get('pb')
        peg = fund.get('peg')
        rev_growth = fund.get('rev_growth')
        if fund.get('yield'): dy = fund.get('yield') * 100
        if rev_growth: rev_growth = rev_growth * 100
        
        # C. TEJ 覆蓋
        if tej_data and code in tej_data:
            t_row = tej_data[code]
            for k, v in t_row.items():
                if '本益比' in k or 'PE' in k: pe = float(v) if v != '-' else pe
                if '淨值比' in k or 'PB' in k: pb = float(v) if v != '-' else pb
                if '殖利率' in k or 'Yield' in k: dy = float(v) if v != '-' else dy
                if '營收成長' in k or 'Growth' in k: rev_growth = float(v) if v != '-' else rev_growth
                if '法人' in k or 'Chips' in k: chips = float(v) if v != '-' else chips

        # D. 合成 PEG
        if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
            peg = calculate_synthetic_peg(pe, rev_growth/100)
            
        # 簡單產業判斷
        industry = 'General'
        if code in ['2330', '2454', '2303', '3034', '3035', '2379']: industry = 'Semicon'
        elif code.startswith('28'): industry = 'Finance'
        elif code in ['2501', '2505', '5522']: industry = 'Construction'

        # 只要有代號就加入，就算沒股價 (會顯示 NaN)
        results.append({
            '代號': code, '名稱': name, 'full_symbol': symbol,
            'close_price': price,
            'pe': pe, 'pb': pb, 'yield': dy,
            'rev_growth': rev_growth, 'peg': peg, 'chips': chips,
            'priceToMA60': ma_bias, 'volatility': volatility,
            'industry': industry
        })
        
    return pd.DataFrame(results)

# --- 7. 熵值模型 ---
def get_entropy_config(industry):
    # 通用配置
    config = {
        'Price vs MA60': {'col': 'priceToMA60', 'dir': 'min', 'w': 1, 'cat': '動能'},
        'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 1, 'cat': '風險'},
        'P/E': {'col': 'pe', 'dir': 'min', 'w': 1, 'cat': '價值'},
        'P/B': {'col': 'pb', 'dir': 'min', 'w': 1, 'cat': '價值'},
    }
    
    if industry == 'Semicon': # 成長型
        config['Rev Growth'] = {'col': 'rev_growth', 'dir': 'max', 'w': 2, 'cat': '成長'}
        config['PEG'] = {'col': 'peg', 'dir': 'min', 'w': 1.5, 'cat': '成長'}
    elif industry == 'Finance': # 價值型
        config['Yield'] = {'col': 'yield', 'dir': 'max', 'w': 2, 'cat': '價值'}
    else: # 一般
        config['Rev Growth'] = {'col': 'rev_growth', 'dir': 'max', 'w': 1, 'cat': '成長'}
        config['Yield'] = {'col': 'yield', 'dir': 'max', 'w': 1, 'cat': '價值'}
        
    return config

def calculate_score(df):
    if df.empty: return df, None
    df_norm = df.copy()
    scores = []
    plans = []
    
    # 填補空值以進行計算 (只影響分數，不影響顯示)
    fill_map = {
        'pe': 50, 'pb': 5, 'yield': 0, 'rev_growth': 0, 'peg': 5, 
        'priceToMA60': 0, 'volatility': 0.5
    }
    calc_df = df.fillna(fill_map)

    for idx, row in calc_df.iterrows():
        config = get_entropy_config(row['industry'])
        total_score = 0
        total_weight = 0
        
        for name, setting in config.items():
            val = row.get(setting['col'])
            all_vals = calc_df[setting['col']]
            
            # 排名百分位
            rank = all_vals.rank(pct=True).get(idx, 0.5)
            if setting['dir'] == 'max': norm = rank
            else: norm = 1 - rank
            
            # 存入 df_norm
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm * 100
            
            total_score += norm * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final, 1))
        
        # 策略判斷
        rev = row.get('rev_growth', 0)
        peg = row.get('peg', 100)
        ma = row.get('priceToMA60', 0)
        
        if final > 70 and rev > 20:
            plans.append("🚀 爆發成長")
        elif final > 60:
            plans.append("🟡 穩健持有")
        elif ma < -0.1:
            plans.append("🟢 超跌反彈")
        else:
            plans.append("⛔ 觀望")
            
    df['Score'] = scores
    df['Strategy'] = plans
    return df.sort_values('Score', ascending=False), df_norm

# --- 8. 繪圖與 AI ---
def get_radar_data(df_norm_row):
    cats = {'價值': 0, '成長': 0, '動能': 0, '風險': 0}
    counts = {'價值': 0, '成長': 0, '動能': 0, '風險': 0}
    for col in df_norm_row.index:
        if col.endswith('_n'):
            cat = col.split('_')[0]
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
        fill='toself', name=title, line_color='#00e676'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                      margin=dict(t=20, b=20, l=20, r=20), height=250)
    return fig

def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = [Paragraph(f"Analysis Report: {stock_data['名稱']}", getSampleStyleSheet()['Heading1'])]
    for k, v in stock_data.items():
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
    except: return "AI 連線失敗。"

AI_PROMPT = """
請針對 [STOCK] 進行投資分析，重點在於「未來半年的爆發力」與「下檔風險」。
數據：PE=[PE], PEG=[PEG], 營收成長=[REV]%, 波動率=[VOL]%.
請給出操作建議 (買進/觀望/賣出) 與關鍵觀察點。
"""

# --- 9. 主程式 ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    
    # 恢復選單功能
    st.markdown("### 1️⃣ 匯入數據")
    uploaded = st.file_uploader("📂 上傳 TEJ (選填)", type=['csv','xlsx'])
    if uploaded: 
        st.session_state['tej_data'] = process_tej_upload(uploaded)
        st.success(f"已載入 TEJ 數據")

    st.markdown("### 2️⃣ 選股模式")
    scan_mode = st.radio("模式選擇", ["🔥 熱門策略", "🏭 產業掃描", "⌨️ 自訂輸入"])
    
    target_stocks = []
    if scan_mode == "⌨️ 自訂輸入":
        # 恢復自訂輸入
        default = ["2330.TW 台積電", "2317.TW 鴻海", "2454.TW 聯發科"]
        selected = st.multiselect("選擇股票", sorted(list(stock_map.values())), default=[s for s in default if s in stock_map.values()])
        manual = st.text_input("手動輸入代號 (例如 2603)", "")
        target_stocks = selected
        if manual: target_stocks.append(f"{manual}.TW")
        
    elif scan_mode == "🏭 產業掃描":
        # 恢復產業掃描
        ind_list = sorted(list(industry_map.keys()))
        ind = st.selectbox("選擇產業", ind_list)
        if ind:
            codes = industry_map[ind]
            target_stocks = [stock_map[c] for c in codes if c in stock_map]
            
    else:
        # 熱門策略
        strat = st.selectbox("策略集", ["台灣50", "AI供應鏈", "高股息"])
        if strat == "台灣50": target_stocks = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海", "2308.TW 台達電", "2881.TW 富邦金"]
        elif strat == "AI供應鏈": target_stocks = ["2330.TW", "2382.TW", "3231.TW", "3017.TW", "3661.TW"]
        else: target_stocks = ["2301.TW", "0056.TW", "2886.TW", "1101.TW"]

    st.info(f"已鎖定 {len(target_stocks)} 檔標的")
    
    if st.button("🚀 啟動全自動掃描", type="primary"):
        st.session_state['scan_finished'] = False
        with st.spinner("正在進行分離式數據抓取 (確保數據完整性)..."):
            raw = batch_scan_stocks_v2(target_stocks, st.session_state['tej_data'])
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 21.0")
    st.caption("Hybrid Data Fetch + Full UI Restored")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    
    if df.empty:
        st.error("❌ 查無數據，請檢查代號或網路。")
    else:
        final_df, df_norm = calculate_score(df)
        
        st.subheader("🏆 潛力標的排行")
        st.dataframe(
            final_df[['代號', '名稱', 'close_price', 'Score', 'Strategy', 'pe', 'rev_growth', 'peg', 'yield', 'chips']],
            column_config={
                "Score": st.column_config.ProgressColumn("綜合戰力", min_value=0, max_value=100, format="%.1f"),
                "close_price": st.column_config.NumberColumn("股價", format="$%.1f"),
                "pe": st.column_config.NumberColumn("本益比"),
                "rev_growth": st.column_config.NumberColumn("營收成長", format="%.2f%%"),
                "peg": st.column_config.NumberColumn("PEG"),
                "chips": st.column_config.NumberColumn("法人籌碼(TEJ)"),
            },
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🎯 深度戰略分析 (UI 已恢復)")
        
        for idx, row in final_df.head(5).iterrows(): # 顯示前5名避免過長
            with st.container():
                st.markdown(f"<div class='stock-card'><h3>{row['名稱']} ({row['代號']}) <span style='font-size:0.8em;color:#00e676'>{row['Strategy']}</span></h3>", unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 2])
                
                if idx in df_norm.index:
                    radar_data = get_radar_data(df_norm.loc[idx])
                    with c1:
                        st.plotly_chart(plot_radar_chart_ui(row['名稱'], radar_data), use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                    - **成長性**: 營收成長 {row.get('rev_growth', 'N/A')}% | PEG {row.get('peg', 'N/A')}
                    - **價值面**: 本益比 {row.get('pe', 'N/A')} | 殖利率 {row.get('yield', 'N/A')}%
                    - **風險面**: 波動率 {row.get('volatility', 0)*100:.1f}% | 季線乖離 {row.get('priceToMA60', 0)*100:.1f}%
                    """)
                    
                    b1, b2 = st.columns(2)
                    if b1.button(f"✨ AI 分析", key=f"ai_{idx}"):
                        p_txt = AI_PROMPT.replace("[STOCK]", row['名稱']).replace("[PE]", str(row.get('pe'))).replace("[PEG]", str(row.get('peg'))).replace("[REV]", str(row.get('rev_growth'))).replace("[VOL]", str(round(row.get('volatility',0)*100,1)))
                        an = call_ai(p_txt)
                        st.markdown(f"<div class='ai-header'>🤖 AI 觀點</div>{an}", unsafe_allow_html=True)
                        
                    pdf = create_pdf(row.to_dict())
                    b2.download_button("📥 下載報告", pdf, f"{row['代號']}.pdf", key=f"dl_{idx}")
                
                st.markdown("</div>", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動全自動掃描」開始。")
