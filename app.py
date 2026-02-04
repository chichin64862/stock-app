import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import io
import time
import requests
import os
import re
from datetime import datetime
from math import pi

# --- PDF 生成庫 (必備) ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
except ImportError:
    st.error("⚠️ 缺少 reportlab 套件。請在 requirements.txt 中加入 `reportlab`")
    st.stop()

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="熵值決策選股及AI深度分析平台 (Ultimate)", 
    page_icon="🔥", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, li { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
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

# --- 6. 核心數據引擎 (Yahoo Deep Fetch) ---
def get_stock_fundamentals(symbol):
    """從 Yahoo Finance 抓取完整財務數據"""
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): 
            symbol += '.TW'
        
        ticker = yf.Ticker(symbol)
        info = ticker.info 
        
        # 提取關鍵數據
        data = {
            'close_price': info.get('currentPrice') or info.get('previousClose'),
            'pe': info.get('trailingPE'),
            'peg': info.get('pegRatio'),
            'pb': info.get('priceToBook'),
            'rev_growth': info.get('revenueGrowth'), # 0.25 = 25%
            'yield': info.get('dividendYield'), # 0.03 = 3%
            'sector': info.get('sector', 'Unknown'),
            'beta': info.get('beta')
        }
        return data
    except Exception:
        return None

def calculate_synthetic_peg(pe, growth_rate):
    """計算合成 PEG: PE / (Growth * 100)"""
    # 成長率要是正的才有意義
    if pe and growth_rate and growth_rate > 0:
        return pe / (growth_rate * 100)
    return None

def process_tej_upload(uploaded_file):
    """處理 TEJ 上傳"""
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

# --- 7. 批量掃描 ---
@st.cache_data(ttl=600, show_spinner=False)
def batch_scan_stocks(stock_list, tej_data=None):
    results = []
    columns = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'rev_growth', 'peg', 'chips', 'industry', 'beta']

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_fundamentals, s.split(' ')[0]): s for s in stock_list}
        
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            try:
                code = stock_str.split(' ')[0].split('.')[0]
                name = stock_str.split(' ')[1] if len(stock_str.split(' ')) > 1 else code
                y_data = future.result()
                
                price = np.nan; pe = np.nan; pb = np.nan; dy = np.nan
                rev_growth = np.nan; peg = np.nan; chips = 0; beta = 1.0
                
                if y_data:
                    price = y_data.get('close_price')
                    pe = y_data.get('pe')
                    pb = y_data.get('pb')
                    if y_data.get('yield'): dy = y_data.get('yield') * 100 
                    if y_data.get('rev_growth'): rev_growth = y_data.get('rev_growth') * 100
                    peg = y_data.get('peg')
                    beta = y_data.get('beta')
                
                # TEJ 覆蓋
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    for k, v in t_row.items():
                        if '法人' in k or 'Chips' in k: chips = float(v) if v != '-' else 0

                # 自動補算 PEG
                if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
                    peg = calculate_synthetic_peg(pe, rev_growth/100)

                if not pd.isna(price):
                    # 簡單產業判斷
                    industry = 'General'
                    if code in ['2330', '2454', '2303', '3034', '3035', '2379']: industry = 'Semicon'
                    elif code.startswith('28'): industry = 'Finance'
                    elif code in ['2501', '2505', '5522']: industry = 'Construction'
                    
                    results.append({
                        '代號': code, '名稱': name, 'close_price': price,
                        'pe': pe, 'pb': pb, 'yield': dy,
                        'rev_growth': rev_growth, 'peg': peg,
                        'chips': chips, 'industry': industry, 'beta': beta
                    })
            except: continue
    
    if not results: return pd.DataFrame(columns=columns)
    return pd.DataFrame(results)

# --- 8. 熵值模型與權重 (動態產業) ---
def get_entropy_config(industry):
    # 預設
    config = {
        'P/E': {'col': 'pe', 'dir': 'min', 'w': 1, 'cat': '估值'},
        'P/B': {'col': 'pb', 'dir': 'min', 'w': 1, 'cat': '估值'},
        'Yield': {'col': 'yield', 'dir': 'max', 'w': 1, 'cat': '財報'},
        'Growth': {'col': 'rev_growth', 'dir': 'max', 'w': 1, 'cat': '成長'},
    }
    # 產業客製化
    if industry == 'Semicon': 
        config['Growth']['w'] = 2.0
        config['PEG'] = {'col': 'peg', 'dir': 'min', 'w': 1.5, 'cat': '成長'}
    elif industry == 'Finance':
        config['Yield']['w'] = 2.0
        config['P/B']['w'] = 1.5
    
    return config

def calculate_score(df):
    if df.empty: return df, None
    df_norm = df.copy()
    scores = []
    plans = []
    
    # 數據補值 (避免計算錯誤)
    for col in ['pe', 'pb', 'yield', 'rev_growth', 'peg']:
        if col in df.columns:
            if col in ['pe', 'pb', 'peg']: fill_val = df[col].max() # 爛的補最大
            else: fill_val = df[col].min() # 爛的補最小
            df[col] = df[col].fillna(fill_val)

    for idx, row in df.iterrows():
        config = get_entropy_config(row.get('industry', 'General'))
        total_score = 0
        total_weight = 0
        
        for name, setting in config.items():
            val = row.get(setting['col'])
            all_vals = df[setting['col']]
            
            # 排名百分位 (0~1)
            rank = all_vals.rank(pct=True).get(idx, 0.5)
            
            if setting['dir'] == 'max': norm_score = rank
            else: norm_score = 1 - rank
            
            # 存入 df_norm 供雷達圖使用
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm_score * 100
            
            total_score += norm_score * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final, 1))
        
        # 戰略判斷
        rev = row.get('rev_growth', 0)
        peg = row.get('peg', 100)
        
        if final > 70 and rev > 20:
            if peg < 1.2: plans.append("🚀 爆發成長 (Strong Buy)")
            else: plans.append("🔥 動能強勢 (Momentum)")
        elif final > 60:
            plans.append("🟡 穩健持有 (Hold)")
        else:
            plans.append("⛔ 觀望 (Wait)")
            
    df['Score'] = scores
    df['Strategy'] = plans
    return df.sort_values('Score', ascending=False), df_norm

# --- 9. 繪圖與 PDF 函數 (已恢復) ---
def get_radar_data(df_norm_row):
    # 從 df_norm 提取各維度分數
    categories = {'估值': 0, '成長': 0, '財報': 0, '籌碼': 0, '技術': 0}
    counts = {'估值': 0, '成長': 0, '財報': 0, '籌碼': 0, '技術': 0}
    
    for col in df_norm_row.index:
        if col.endswith('_n'):
            cat = col.split('_')[0]
            if cat in categories:
                categories[cat] += df_norm_row[col]
                counts[cat] += 1
                
    # 取平均
    final_radar = {}
    for k, v in categories.items():
        if counts[k] > 0: final_radar[k] = v / counts[k]
        else: final_radar[k] = 50 # 預設中立
    return final_radar

def plot_radar_chart_ui(title, radar_data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(radar_data.values()), theta=list(radar_data.keys()),
        fill='toself', name=title, line_color='#00e676'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                      margin=dict(t=20, b=20, l=20, r=20), height=250)
    return fig

# --- 10. AI Prompt ---
AI_PROMPT = """
你現在是華爾街頂尖的成長型基金經理人。請針對 [STOCK] 撰寫投資分析報告。
重點關注：
1. **成長性驗證**：營收成長 (YoY) 是否加速？PEG 是否 < 1.5 (合理價格買成長)？
2. **護城河與風險**：產業地位與潛在的灰犀牛風險。
3. **操作建議**：給出未來 6-12 個月的目標價位區間與操作策略。
數據參考：
[DATA]
"""

def call_ai(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers=headers, json=data)
        return r.json()['candidates'][0]['content']['parts'][0]['text']
    except: return "AI 分析連線失敗，請稍後再試。"

# --- 11. PDF 輸出 ---
def create_pdf(stock_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = [Paragraph("AlphaCore 深度分析報告", getSampleStyleSheet()['Heading1'])]
    for k, v in stock_data.items():
        story.append(Paragraph(f"{k}: {v}", getSampleStyleSheet()['Normal']))
    try: doc.build(story)
    except: pass
    buffer.seek(0)
    return buffer

# --- 12. 主程式 ---
with st.sidebar:
    st.title("🎛️ 決策控制台")
    with st.expander("📂 匯入 TEJ (選填)"):
        uploaded = st.file_uploader("上傳 CSV/Excel", type=['csv','xlsx'])
        if uploaded: 
            st.session_state['tej_data'] = process_tej_upload(uploaded)
            st.success("TEJ 數據已載入")
            
    strategy = st.selectbox("選股策略", ["台灣50", "AI供應鏈", "高股息"])
    if strategy == "台灣50": targets = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海"]
    elif strategy == "AI供應鏈": targets = ["2330.TW 台積電", "2382.TW 廣達", "3231.TW 緯創"]
    else: targets = ["2301.TW 光寶科", "0056.TW 高股息"]
    
    if st.button("🚀 啟動全自動掃描", type="primary"):
        with st.spinner("正在挖掘 Yahoo 財報數據..."):
            raw = batch_scan_stocks(targets, st.session_state['tej_data'])
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 20.0")
    st.caption("Yahoo Deep Fetch + Dynamic Sector Weighting + Full UI Restored")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    
    if df.empty:
        st.error("❌ 查無數據，請稍後再試。")
    else:
        # 計算與排名
        final_df, df_norm = calculate_score(df)
        
        # 1. 總表
        st.subheader("🏆 潛力標的排行")
        st.dataframe(final_df[['代號', '名稱', 'close_price', 'Score', 'Strategy', 'pe', 'rev_growth', 'peg']], use_container_width=True)
        
        # 2. 個股卡片 (UI 回歸！)
        st.markdown("---")
        st.subheader("🎯 深度戰略分析")
        
        for idx, row in final_df.iterrows():
            with st.container():
                st.markdown(f"<div class='stock-card'><h3>{row['名稱']} ({row['代號']}) - {row['Strategy']}</h3>", unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 2])
                
                # 準備數據
                radar_data = get_radar_data(df_norm.loc[idx])
                
                with c1:
                    st.plotly_chart(plot_radar_chart_ui(row['名稱'], radar_data), use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                    - **股價**: {row['close_price']}
                    - **本益比**: {row['pe']}
                    - **營收成長**: {row['rev_growth']:.2f}%
                    - **PEG**: {row['peg']:.2f} (越低越好)
                    """)
                    
                    # 按鈕區
                    b1, b2 = st.columns(2)
                    if b1.button(f"✨ AI 分析 {row['名稱']}", key=f"ai_{idx}"):
                        data_ctx = f"PE={row['pe']}, PEG={row['peg']}, RevGrowth={row['rev_growth']}%"
                        prompt = AI_PROMPT.replace("[STOCK]", row['名稱']).replace("[DATA]", data_ctx)
                        analysis = call_ai(prompt)
                        st.markdown(f"<div class='ai-header'>🤖 AI 觀點</div>{analysis}", unsafe_allow_html=True)
                        
                    pdf_data = create_pdf(row.to_dict())
                    b2.download_button(f"📥 下載報告", pdf_data, file_name=f"{row['代號']}_report.pdf", key=f"pdf_{idx}")
                
                st.markdown("</div>", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動全自動掃描」開始。")
