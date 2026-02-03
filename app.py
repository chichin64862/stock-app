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
import os
import io
import re
import random
from datetime import datetime
import matplotlib.pyplot as plt
from math import pi

# --- PDF 生成庫檢查 ---
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
    page_title="熵值決策選股及AI深度分析平台", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 針對性修復 ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, li { color: #e6e6e6 !important; font-family: 'Roboto', sans-serif; }
    div[role="menu"] div, div[role="menu"] span, div[role="menu"] label { color: #31333F !important; font-weight: 500 !important; }
    div[role="menu"] label { color: #31333F !important; }
    div[data-baseweb="select"] > div { background-color: #262730 !important; border-color: #4b4b4b !important; color: white !important; }
    div[data-baseweb="popover"], ul[data-baseweb="menu"] { background-color: #ffffff !important; border: 1px solid #cccccc !important; }
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] div, li[role="option"] { color: #000000 !important; font-weight: 500 !important; }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #238636 !important; color: #ffffff !important; }
    .stDownloadButton button { background-color: #1f2937 !important; color: #ffffff !important; border: 1px solid #238636 !important; white-space: nowrap !important; min-width: 180px !important; display: flex !important; align-items: center !important; justify-content: center !important; }
    .stDownloadButton button:hover { border-color: #58a6ff !important; color: #58a6ff !important; }
    .stDownloadButton p { color: inherit !important; font-size: 1rem !important; }
    [data-testid="stElementToolbar"] { background-color: #262730 !important; border: 1px solid #4b4b4b !important; }
    [data-testid="stElementToolbar"] svg { fill: #ffffff !important; color: #ffffff !important; }
    [data-testid="stElementToolbar"] button:hover { background-color: #4b4b4b !important; }
    input { color: #ffffff !important; caret-color: #ffffff !important; background-color: #262730 !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stock-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 15px; }
    .pdf-center { background-color: #1f2937; padding: 20px; border-radius: 8px; border-left: 5px solid #238636; margin-bottom: 20px; }
    .ai-header { color: #58a6ff !important; font-weight: bold; font-size: 1.3rem; margin-bottom: 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'df_norm' not in st.session_state: st.session_state['df_norm'] = None
if 'market_fundamentals' not in st.session_state: st.session_state['market_fundamentals'] = {}

# --- 4. API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！請確認您已在 Streamlit Cloud > Settings > Secrets 中設定 `GEMINI_API_KEY`。")
    st.stop()

# --- 5. 環境設定 ---
proxies = {}
if os.getenv("HTTP_PROXY"): proxies["http"] = os.getenv("HTTP_PROXY")
if os.getenv("HTTPS_PROXY"): proxies["https"] = os.getenv("HTTPS_PROXY")

# --- 6. 字型下載 ---
@st.cache_resource
def register_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    url = "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get(url, allow_redirects=True, timeout=10)
            if r.status_code == 200:
                with open(font_path, 'wb') as f: f.write(r.content)
            else: return False
        except: return False
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
            return True
    except: return False
    return False

font_ready = register_chinese_font()

# --- 7. 圖表繪製 ---
def generate_radar_img_mpl(radar_data):
    try:
        categories = list(radar_data.keys())
        values = [v if not pd.isna(v) else 0 for v in radar_data.values()]
        values += values[:1]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#00e676')
        ax.fill(angles, values, '#00e676', alpha=0.25)
        plt.xticks(angles[:-1], categories, color='black', size=10)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75], ["25", "50", "75"], color="grey", size=7)
        plt.ylim(0, 100)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close(fig)
        return buf
    except: return None

def generate_trend_img_mpl(full_symbol, ma_bias):
    try:
        stock_hist = yf.Ticker(full_symbol).history(period="6mo")
        if stock_hist.empty: return None
        dates = stock_hist.index
        prices = stock_hist['Close']
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(dates, prices, color='#29b6f6', linewidth=2)
        ax.scatter(dates[-1], prices.iloc[-1], color='#00e676', s=50, zorder=5)
        if pd.isna(ma_bias): ma_bias = 0
        trend_status = "Overheated" if ma_bias > 0.15 else ("Value Zone" if ma_bias < -0.05 else "Momentum")
        ax.set_title(f"Trend: {trend_status}", color='black', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close(fig)
        return buf
    except: return None

def plot_radar_chart_ui(row_name, radar_data):
    clean_data = {k: (v if not pd.isna(v) else 0) for k, v in radar_data.items()}
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(clean_data.values()), theta=list(clean_data.keys()),
        fill='toself', name=row_name, line_color='#00e676', fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='#8b949e'), bgcolor='rgba(0,0,0,0)'),
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e6e6e6', size=12), height=250
    )
    return fig

def plot_trend_chart_ui(full_symbol, ma_bias):
    try:
        stock_hist = yf.Ticker(full_symbol).history(period="6mo")
        if stock_hist.empty:
            try:
                code = full_symbol.split('.')[0]
                ts = twstock.Stock(code)
                data = ts.fetch_31()
                if data:
                    dates = [d.date for d in data]
                    prices = [d.close for d in data]
                    stock_hist = pd.DataFrame({'Close': prices}, index=dates)
            except: pass
        if stock_hist.empty: return None
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['Close'], mode='lines', name='Price', line=dict(color='#29b6f6', width=2)))
        last_price = stock_hist['Close'].iloc[-1]
        fig_trend.add_trace(go.Scatter(x=[stock_hist.index[-1]], y=[last_price], mode='markers', marker=dict(color='#00e676', size=10), name='Current'))
        if pd.isna(ma_bias): ma_bias = 0
        timing_msg = "Value Zone" if ma_bias < -0.05 else "Momentum"
        if ma_bias > 0.15: timing_msg = "Overheated"
        fig_trend.update_layout(
            title=dict(text=timing_msg, font=dict(size=14, color='#e6e6e6')),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30363d'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=30,b=0), height=250, showlegend=False,
            font=dict(color='#e6e6e6')
        )
        return fig_trend
    except: return None

# --- 8. PDF 生成引擎 ---
def create_pdf(stock_data_list):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    font_name = 'ChineseFont' if font_ready else 'Helvetica'
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=22, spaceAfter=20, alignment=1, textColor=colors.HexColor("#2C3E50"))
    h2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontName=font_name, fontSize=16, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor("#2980B9"))
    h3_style = ParagraphStyle('Heading3', parent=styles['Heading3'], fontName=font_name, fontSize=12, spaceBefore=10, textColor=colors.HexColor("#16A085"))
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=16, spaceAfter=5)
    
    story.append(Paragraph(f"熵值決策選股及AI深度分析報告", title_style))
    story.append(Paragraph(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')} (僅供參考使用)", normal_style))
    story.append(Spacer(1, 20))

    for idx, stock in enumerate(stock_data_list):
        if idx > 0: story.append(PageBreak()) 
        name = stock['name']
        story.append(Paragraph(f"🎯 {name}", h2_style))
        story.append(Paragraph("_" * 60, normal_style))
        story.append(Spacer(1, 10))
        
        action = stock.get('action', 'N/A')
        story.append(Paragraph(f"⚡ 系統戰略指令: <b>{action}</b>", h3_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("📊 核心數據概覽 (Key Metrics)", h3_style))
        peg = stock.get('peg', 'N/A')
        if pd.isna(peg) or str(peg) == 'nan': peg = 'N/A'
        
        # 顯示欄位修正：因使用官方數據，PE與殖利率更準確
        pe_display = stock.get('pe', 'N/A')
        yield_display = stock.get('fcf_yield', 'N/A') # 用殖利率替代 FCF 顯示
        
        t_data = [
            ["指標", "數值", "指標", "數值"],
            [f"收盤價", f"{stock['price']}", f"Entropy Score", f"{stock['score']}"],
            [f"本益比 (P/E)", f"{pe_display}", f"季線乖離", f"{stock.get('ma_bias', 'N/A')}"],
            [f"股價淨值比 (P/B)", f"{stock.get('pb', 'N/A')}", f"殖利率 (Yield)", f"{yield_display}"],
            [f"合成 ROE", f"{stock.get('roe_syn', 'N/A')}", f"Beta", f"{stock.get('beta', 'N/A')}"],
        ]
        t = Table(t_data, colWidths=[100, 130, 100, 130])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        radar = stock.get('radar_data', {})
        try: ma_bias_val = float(stock.get('ma_bias', '0').strip('%')) / 100
        except: ma_bias_val = 0
        full_symbol = stock.get('full_symbol', '')
        
        charts_row = []
        radar_buf = generate_radar_img_mpl(radar)
        if radar_buf: charts_row.append(Image(radar_buf, width=200, height=200))
        trend_buf = generate_trend_img_mpl(full_symbol, ma_bias_val)
        if trend_buf: charts_row.append(Image(trend_buf, width=250, height=150))
            
        if charts_row:
            story.append(Paragraph("📈 戰略因子與趨勢分析", h3_style))
            col_w = 460 / len(charts_row)
            c_table = Table([charts_row], colWidths=[col_w] * len(charts_row))
            c_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
            story.append(c_table)
            story.append(Spacer(1, 10))

        analysis = stock.get('analysis')
        if analysis:
            story.append(Paragraph("🤖 AI 深度投資建議", h3_style))
            formatted = analysis.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted)
            formatted = formatted.replace("\n", "<br/>").replace("### ", "").replace("## ", "").replace("# ", "")
            story.append(Paragraph(formatted, normal_style))
        else:
            story.append(Paragraph("💡 (此份報告僅包含量化數據，尚未執行 AI 深度解讀)", normal_style))
            
    try: doc.build(story)
    except Exception as e:
        buffer = io.BytesIO()
        c = SimpleDocTemplate(buffer)
        story = [Paragraph(f"PDF Error: {str(e)}", getSampleStyleSheet()['Normal'])]
        c.build(story)
    buffer.seek(0)
    return buffer

# --- 9. Gemini API ---
def get_available_model(key):
    default_model = "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, proxies=proxies, timeout=5, verify=False)
        if response.status_code == 200:
            data = response.json()
            for m in data.get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'flash' in m['name']: return m['name'].replace('models/', '')
    except: pass
    return default_model

def call_gemini_api(prompt):
    target_model = get_available_model(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    try:
        response = requests.post(url, headers=headers, json=data, proxies=proxies, timeout=60, verify=False)
        if response.status_code == 200: return response.json()['candidates'][0]['content']['parts'][0]['text']
        else: return f"❌ 分析失敗 (Code {response.status_code})"
    except Exception as e: return f"❌ 連線逾時或錯誤: {str(e)}"

HEDGE_FUND_PROMPT = """
【指令】
請針對 **[STOCK]** 撰寫一份客觀的「投資決策分析報告」。

【⚠️ 分析邏輯指令】
請直接根據量化數據與產業現況進行分析，無需扮演任何角色或提及任何機構名稱。報告內容應包含：

1. **財務健康度評估**：
   - 結合「本益比」與「股價淨值比」判斷估值位階。
   - 透過「殖利率」評估現金回饋能力。

2. **營收與成長動能**：
   - 根據「合成 ROE (P/B 除以 P/E)」推算資本回報效率。
   - 指出目前是處於「價值低估」、「合理評價」還是「成長溢價」階段。

3. **操作建議與風險提示**：
   - **投資評等**：請給出 [強力買進 / 區間操作 / 減持觀望] 建議。
   - **關鍵點位**：設定合理的「防禦區間 (Support)」與「目標區間 (Target)」。
   - **觀察指標**：列出未來最需要關注的一個風險變數。

【最新市場即時數據】
[DATA_CONTEXT]
"""

# --- 10. 數據處理 ---
@st.cache_data
def get_tw_stock_info():
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

# 調整指標配置：使用官方數據欄位
indicators_config = {
    'Price vs MA60': {'col': 'priceToMA60', 'direction': '負向', 'name': '季線乖離', 'category': '技術'},
    'Volume Change': {'col': 'volumeRatio', 'direction': '正向', 'name': '量能比', 'category': '籌碼'},
    'P/E Ratio': {'col': 'pe', 'direction': '負向', 'name': '本益比', 'category': '估值'}, # 取代 PEG
    'P/B Ratio': {'col': 'pb', 'direction': '負向', 'name': '股價淨值比', 'category': '估值'},
    'Synthetic ROE': {'col': 'roe_syn', 'direction': '正向', 'name': '合成ROE', 'category': '財報'}, # 取代 ROE
    'Dividend Yield': {'col': 'yield', 'direction': '正向', 'name': '殖利率', 'category': '財報'}, # 取代 FCF
}

# --- 【核心】TWSE/TPEX 官方開放數據連接器 ---
@st.cache_data(ttl=3600)
def fetch_market_fundamentals():
    """下載全市場個股本益比、殖利率、股價淨值比 (官方 Open Data)"""
    market_data = {}
    
    # 1. 上市 (TWSE)
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        r = requests.get(url_twse, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data:
                code = item['Code']
                try:
                    pe = float(item['PEratio']) if item['PEratio'] != "-" else 0
                    pb = float(item['PBratio']) if item['PBratio'] != "-" else 0
                    dy = float(item['DividendYield']) if item['DividendYield'] != "-" else 0
                    market_data[code] = {'pe': pe, 'pb': pb, 'yield': dy}
                except: pass
    except: pass
    
    # 2. 上櫃 (TPEX)
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
        r = requests.get(url_tpex, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data:
                code = item['SecuritiesCompanyCode']
                try:
                    pe = float(item['PERatio']) if item['PERatio'] != "-" else 0
                    pb = float(item['PBRatio']) if item['PBRatio'] != "-" else 0
                    dy = float(item['DividendYield']) if item['DividendYield'] != "-" else 0
                    market_data[code] = {'pe': pe, 'pb': pb, 'yield': dy}
                except: pass
    except: pass
            
    return market_data

def get_radar_data(df_norm_row, config):
    categories = {'技術': [], '籌碼': [], '財報': [], '估值': []}
    for key, cfg in config.items():
        cat = cfg['category']
        col_n = f"{cfg['col']}_n"
        if col_n in df_norm_row:
            score = df_norm_row[col_n] * 100
            categories[cat].append(score)
    return {k: np.mean(v) if v else 0 for k, v in categories.items()}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hybrid_data(tickers_list):
    results = []
    
    # 1. 獲取官方基本面數據 (Map)
    fund_map = fetch_market_fundamentals()
    
    # 2. 批量獲取 Yahoo 股價 (只抓價格與量，這是 Yahoo 最不容易擋的部分)
    try:
        symbols = [t.split(' ')[0] for t in tickers_list]
        # threads=False 避免被當攻擊
        data = yf.download(symbols, period="3mo", group_by='ticker', progress=False, threads=False)
        
        # 解析數據
        for ticker_full in tickers_list:
            parts = ticker_full.split(' ')
            symbol = parts[0]
            name = parts[1] if len(parts) > 1 else symbol
            code = symbol.split('.')[0]
            
            # 預設值
            price = np.nan
            ma_bias = 0
            vol_ratio = 1.0
            
            # 從 Yahoo Data 提取
            try:
                df = data if len(symbols) == 1 else (data[symbol] if symbol in data else pd.DataFrame())
                if not df.empty and 'Close' in df.columns:
                    latest = df.iloc[-1]
                    price = float(latest['Close'])
                    if not pd.isna(price):
                        # 技術指標
                        ma60 = df['Close'].rolling(window=60).mean().iloc[-1]
                        if not pd.isna(ma60) and ma60 > 0:
                            ma_bias = (price / ma60) - 1
                        
                        vol_curr = df['Volume'].iloc[-1]
                        vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
                        if not pd.isna(vol_avg) and vol_avg > 0:
                            vol_ratio = vol_curr / vol_avg
            except: pass
            
            # 若 Yahoo 失敗，嘗試 TWSE 補價 (救援)
            if pd.isna(price):
                try:
                    realtime = twstock.realtime.get(code)
                    if realtime['success']:
                        p_str = realtime['realtime'].get('latest_trade_price', '-')
                        if p_str and p_str != '-': price = float(p_str)
                except: pass
            
            # 若有價格，進行數據合併
            if not pd.isna(price):
                # 從官方 Map 獲取真實財報數據
                f_data = fund_map.get(code, {'pe': 0, 'pb': 0, 'yield': 0})
                
                # 計算合成 ROE: ROE = P/B / P/E
                # 數學上: E/B = (P/B) / (P/E)
                roe_syn = 0
                if f_data['pe'] > 0 and f_data['pb'] > 0:
                    roe_syn = (f_data['pb'] / f_data['pe']) * 100
                
                results.append({
                    '代號': code,
                    'full_symbol': symbol,
                    '名稱': name,
                    'close_price': price,
                    'priceToMA60': ma_bias, 
                    'volumeRatio': vol_ratio,
                    'pe': f_data['pe'],
                    'pb': f_data['pb'],
                    'yield': f_data['yield'],
                    'roe_syn': roe_syn,
                    'pegRatio': np.nan, # 已被 PE 取代
                    'debtToEquity': np.nan, # 無法獲取，暫不計分
                    'fcfYield': np.nan, # 已被 Yield 取代
                    'beta': 1.0
                })
                
    except Exception as e: pass
            
    return pd.DataFrame(results)

def calculate_entropy_score(df, config):
    if df.empty: return df, None, "數據抓取為空，請檢查代號是否正確。", None
    df_norm = df.copy()
    for key, cfg in config.items():
        col = cfg['col']
        # 容錯：若欄位不存在補 0
        if col not in df.columns: df[col] = 0
        
        # 填補 0 值 (避免官方數據缺漏)
        # 對於 PE，0 通常代表虧損，設為極大值(懲罰)
        if col == 'pe':
            df[col] = df[col].replace(0, 1000) 
            
        if cfg['direction'] == '正向': fill_val = df[col].min()
        else: fill_val = df[col].max()
        
        df[col] = df[col].fillna(fill_val)
        df_norm[col] = df[col]
        
        q_low = df[col].quantile(0.05); q_high = df[col].quantile(0.95)
        df_norm[col] = df[col].clip(lower=q_low, upper=q_high)
        mn, mx = df_norm[col].min(), df_norm[col].max()
        denom = mx - mn
        if denom == 0: df_norm[f'{col}_n'] = 0.5
        else:
            if cfg['direction'] == '正向': df_norm[f'{col}_n'] = (df_norm[col] - mn) / denom
            else: df_norm[f'{col}_n'] = (mx - df_norm[col]) / denom
            
    m = len(df); k = 1 / np.log(m) if m > 1 else 0; weights = {}
    for key, cfg in config.items():
        col = cfg['col']
        if f'{col}_n' in df_norm.columns:
            p = df_norm[f'{col}_n'] / df_norm[f'{col}_n'].sum() if df_norm[f'{col}_n'].sum() != 0 else 0
            e = -k * np.sum(p * np.log(p + 1e-9))
            weights[key] = 1 - e 
    tot = sum(weights.values())
    if tot == 0: fin_w = {k: 1/len(weights) for k in weights}
    else: fin_w = {k: v/tot for k, v in weights.items()}
    df['Score'] = 0
    for key, cfg in config.items():
        if f'{cfg["col"]}_n' in df_norm.columns:
            df['Score'] += fin_w[key] * df_norm[f'{cfg["col"]}_n'] 
    df['Score'] = (df['Score']*100).round(1)
    return df.sort_values('Score', ascending=False), fin_w, None, df_norm

def render_factor_bars(radar_data):
    html = ""
    colors = {'技術': '#29b6f6', '籌碼': '#ab47bc', '財報': '#ffca28', '估值': '#ef5350'}
    for cat, score in radar_data.items():
        color = colors.get(cat, '#8b949e')
        blocks = int(score / 10)
        visual_bar = "■" * blocks + "░" * (10 - blocks)
        html += f"""<div style="margin-bottom: 8px;"><div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#e6e6e6;"><span><span style="color:{color};">●</span> {cat}</span><span>{score:.0f}%</span></div><div style="font-family: monospace; color:{color}; letter-spacing: 2px;">{visual_bar}</div></div>"""
    return html

# --- 12. 主儀表板與流程 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.markdown("---")
    
    if st.button("🔴 清除快取並重置", use_container_width=True):
        st.cache_data.clear()
        if 'raw_data' in st.session_state: del st.session_state['raw_data']
        if 'scan_finished' in st.session_state: del st.session_state['scan_finished']
        st.rerun()
        
    st.markdown("---")
    scan_mode = st.radio("選股模式：", ["🔥 熱門策略掃描", "🏭 產業類股掃描", "自行輸入/多選"], label_visibility="collapsed")
    target_stocks = []
    
    st.caption("🔍 若找不到股票，請直接輸入代號 (如 1802):")
    manual_input = st.text_input("手動輸入代號:", placeholder="例如: 1802 或 2330", label_visibility="collapsed")
    
    if scan_mode == "自行輸入/多選":
        default_selection = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海"]
        selected = st.multiselect("選擇股票:", options=sorted(list(stock_map.values())), default=[s for s in default_selection if s in stock_map.values()])
        target_stocks = selected
    elif scan_mode == "🔥 熱門策略掃描":
        strategy = st.selectbox("策略集:", ["台灣50成份股 (大型權值)", "中型100成份股 (成長潛力)", "高股息熱門股 (存股族)", "AI 供應鏈概念", "貨櫃航運三雄"])
        if strategy == "台灣50成份股 (大型權值)":
            codes = ["2330", "2454", "2317", "2308", "2382", "2303", "2881", "2882", "2891", "1216", "2002", "1301", "1303", "2603", "3008", "3045", "2912", "5880", "2886", "2892", "2207", "1101", "2357", "2395", "3231", "2379", "3034", "2345", "3711", "2885"]
            target_stocks = [f"{c}.TW {stock_map.get(f'{c}.TW', '').split(' ')[-1]}" for c in codes if f"{c}.TW" in stock_map]
        elif strategy == "中型100成份股 (成長潛力)":
            codes = ["2344", "2376", "2383", "2368", "3443", "3661", "3529", "3035", "3037", "3017", "2313", "2324", "2352", "2353", "2356", "2327", "2385", "2408", "2409", "2449", "2451", "2474", "2492", "2498", "2542", "2609", "2610", "2615", "2618"]
            target_stocks = [f"{c}.TW {stock_map.get(f'{c}.TW', '').split(' ')[-1]}" for c in codes if f"{c}.TW" in stock_map]
        elif strategy == "高股息熱門股 (存股族)":
            codes = ["2301", "2324", "2352", "2356", "2382", "2385", "2449", "2454", "2603", "3034", "3037", "3044", "3231", "3702", "3711", "4915", "4938", "4958", "5388", "5483", "6176", "6239", "8131"]
            target_stocks = []
            for c in codes:
                if f"{c}.TW" in stock_map: target_stocks.append(stock_map[f"{c}.TW"])
                elif f"{c}.TWO" in stock_map: target_stocks.append(stock_map[f"{c}.TWO"])
        elif strategy == "AI 供應鏈概念":
            codes = ["2330", "2317", "2382", "3231", "6669", "3443", "3661", "3035", "2376", "2368", "3017", "2301", "2356", "3037", "2308", "2421", "2454", "3034"]
            target_stocks = [f"{c}.TW {stock_map.get(f'{c}.TW', '').split(' ')[-1]}" for c in codes if f"{c}.TW" in stock_map]
        elif strategy == "貨櫃航運三雄":
            target_stocks = ["2603.TW 長榮", "2609.TW 陽明", "2615.TW 萬海"]
            
    elif scan_mode == "🏭 產業類股掃描":
        all_industries = sorted(list(industry_map.keys()))
        selected_industry = st.selectbox("選擇產業:", all_industries)
        if selected_industry:
            codes = industry_map[selected_industry]
            target_stocks = [stock_map[c] for c in codes if c in stock_map]
    
    if manual_input:
        target_stocks.append(manual_input)
            
    st.info(f"已鎖定 {len(target_stocks)} 檔標的")
    st.markdown("---")
    run_btn = st.button("🚀 啟動全自動掃描", type="primary", use_container_width=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股及AI深度分析平台")
    st.caption("Entropy Scoring • Factor Radar • PDF Reporting (僅供參考使用)")
with col2:
    if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
         st.metric("Total Scanned", f"{len(st.session_state['raw_data'])} Stocks", delta="Live Update")

if run_btn:
    if not target_stocks:
        st.warning("⚠️ 請至少選擇一檔股票，或在左側輸入代號 (例如 1802)。")
    else:
        st.session_state['analysis_results'] = {}
        st.session_state['raw_data'] = None
        st.session_state['df_norm'] = None
        
        with st.spinner("🚀 正在啟動雙網架構掃描 (Yahoo 報價 + TWSE/TPEX 官方財報)..."):
            raw = fetch_hybrid_data(target_stocks)
            
        if not raw.empty:
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()
        else:
            st.error("❌ 掃描失敗：所有來源皆無回應，請稍後再試。")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    # 更新：檢查欄位是否包含新的指標
    if 'pe' not in st.session_state['raw_data'].columns:
        st.session_state['raw_data'] = None
        st.rerun()

    raw = st.session_state['raw_data']
    res, w, err, df_norm = calculate_entropy_score(raw, indicators_config)
    
    if err:
        st.error(err)
    else:
        st.session_state['df_norm'] = df_norm 
        
        def get_trend_label(bias):
            if pd.isna(bias): return "⚪ 數據不足"
            if bias < -0.05: return "🟢 超跌/買點"
            elif bias > 0.15: return "🔴 過熱/賣點"
            else: return "🟡 盤整/持有"
            
        def determine_action_plan(row):
            score = row['Score']
            bias = row['priceToMA60']
            if pd.isna(bias): bias = 0 
            
            if score >= 75:
                if bias < -0.05: return "🚀 強力抄底 (Deep Value Buy)"
                elif bias > 0.15: return "👀 拉回買進 (Buy on Dip)"
                else: return "🔥 強力買進 (Strong Buy)"
            elif score >= 50:
                if bias < -0.1: return "🟢 超跌反彈 (Rebound)"
                elif bias > 0.2: return "🔴 高檔調節 (Take Profit)"
                else: return "🟡 持有續抱 (Hold)"
            else:
                return "⛔ 觀望/賣出 (Avoid/Sell)"
        
        res['Trend'] = res['priceToMA60'].apply(get_trend_label)
        res['Action Plan'] = res.apply(determine_action_plan, axis=1)
        top_n = 10
        top_stocks = res.head(top_n)

        st.markdown("### 🏆 Top 10 潛力標的 (Entropy Ranking)")
        st.dataframe(
            top_stocks[['代號', '名稱', 'close_price', 'Score', 'pe', 'priceToMA60', 'pb', 'yield', 'Action Plan']],
            column_config={
                "Score": st.column_config.ProgressColumn("Entropy Score", format="%.1f", min_value=0, max_value=100),
                "close_price": st.column_config.NumberColumn("Price", format="%.2f"),
                "pe": st.column_config.NumberColumn("P/E (本益比)", format="%.2f"),
                "priceToMA60": st.column_config.NumberColumn("MA Bias", format="%.2%"),
                "pb": st.column_config.NumberColumn("P/B (淨值比)", format="%.2f"),
                "yield": st.column_config.NumberColumn("Yield (殖利率)", format="%.2f%%"),
                "Action Plan": st.column_config.TextColumn("戰略指令 (Strategy)"),
            },
            hide_index=True, use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 📥 戰略報告下載中心 (All-in-One Reports)")
        
        with st.container():
            st.markdown('<div class="pdf-center">', unsafe_allow_html=True)
            
            if len(res) > 0:
                col_info, col_dl = st.columns([0.65, 0.35], vertical_alignment="center")
                with col_info:
                    st.success(f"✅ 已準備 {len(res)} 份量化數據報告。點擊 AI 分析後，內容將自動更新。")
                with col_dl:
                    bulk_data_final = []
                    for idx, row in res.iterrows():
                        stock_name = f"{row['代號']} {row['名稱']}"
                        if idx in df_norm.index:
                            norm_row = df_norm.loc[idx]
                            radar = get_radar_data(norm_row, indicators_config)
                            analysis_text = st.session_state['analysis_results'].get(stock_name, None)
                            
                            bulk_data_final.append({
                                'name': stock_name,
                                'price': row['close_price'],
                                'score': row['Score'],
                                'pe': row.get('pe', 0),
                                'pb': row.get('pb', 0),
                                'fcf_yield': f"{row.get('yield', 0):.2f}%",
                                'roe_syn': f"{row.get('roe_syn', 0):.2f}%",
                                'ma_bias': f"{row['priceToMA60']:.2%}",
                                'radar_data': radar,
                                'analysis': analysis_text,
                                'action': row['Action Plan'],
                                'full_symbol': row['full_symbol']
                            })
                    
                    if bulk_data_final:
                        pdf_data_final = create_pdf(bulk_data_final)
                        st.download_button(
                            label="📑 下載全部報告 (PDF)",
                            data=pdf_data_final,
                            file_name=f"AlphaCore_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 深度戰略分析 (Strategic Deep Dive)")
        
        for i, (index, row) in enumerate(top_stocks.iterrows()):
            stock_name = f"{row['代號']} {row['名稱']}"
            is_analyzed = (stock_name in st.session_state['analysis_results'])
            
            with st.container():
                st.markdown(f"""<div class="stock-card"><h3>{stock_name} <span style="font-size:0.6em;color:#8b949e">NT$ {row['close_price']}</span> <span style="font-size:0.8em;color:#00e676;border:1px solid #00e676;padding:2px 5px;border-radius:4px;margin-left:10px;">{row['Action Plan']}</span></h3>""", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1.5, 1.2, 2])
                
                if index in df_norm.index:
                    norm_row = df_norm.loc[index]
                    radar_data = get_radar_data(norm_row, indicators_config)
                
                    with c1:
                        fig_radar = plot_radar_chart_ui(row['名稱'], radar_data)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with c2:
                        st.markdown("**因子貢獻解析**")
                        st.markdown(render_factor_bars(radar_data), unsafe_allow_html=True)
                
                with c3:
                    st.markdown("**配置時機判定 (Trend vs Value)**")
                    fig_trend = plot_trend_chart_ui(row['full_symbol'], row['priceToMA60'])
                    if fig_trend:
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.warning("⚠️ 無法取得歷史數據")

                col_btn, col_dl = st.columns([3, 1])
                
                with col_btn:
                     if st.button(f"✨ 生成分析報告", key=f"btn_{i}", use_container_width=True, disabled=is_analyzed):
                         if not is_analyzed:
                            with st.spinner(f"⚡ AI 正在為您撰寫 {stock_name} 的投資備忘錄..."):
                                pe_val = row.get('pe', 0)
                                pb_val = row.get('pb', 0)
                                dy_val = row.get('yield', 0)
                                roe_syn = row.get('roe_syn', 0)
                                
                                real_time_data = f"""
                                - 收盤價: {row['close_price']}
                                - 本益比 (P/E): {pe_val:.2f}
                                - 股價淨值比 (P/B): {pb_val:.2f}
                                - 殖利率 (Yield): {dy_val:.2f}%
                                - 合成 ROE: {roe_syn:.2f}%
                                - 因子得分: {radar_data} (滿分100)
                                - 季線乖離: {row['priceToMA60']:.2%}
                                """
                                final_prompt = HEDGE_FUND_PROMPT.replace("[STOCK]", stock_name).replace("[DATA_CONTEXT]", real_time_data)
                                result = call_gemini_api(final_prompt)
                                st.session_state['analysis_results'][stock_name] = result
                                st.rerun()
                
                with col_dl:
                    single_data = [{
                        'name': stock_name,
                        'price': row['close_price'],
                        'score': row['Score'],
                        'pe': row.get('pe', 0),
                        'pb': row.get('pb', 0),
                        'fcf_yield': f"{row.get('yield', 0):.2f}%",
                        'roe_syn': f"{row.get('roe_syn', 0):.2f}%",
                        'ma_bias': f"{row['priceToMA60']:.2%}",
                        'radar_data': radar_data,
                        'analysis': st.session_state['analysis_results'].get(stock_name, None),
                        'action': row['Action Plan'],
                        'full_symbol': row['full_symbol']
                    }]
                    pdf_data = create_pdf(single_data)
                    st.download_button(
                        label="📥 下載個股 PDF",
                        data=pdf_data,
                        file_name=f"{stock_name}_Report.pdf",
                        mime="application/pdf",
                        key=f"dl_{i}",
                        use_container_width=True
                    )

                if is_analyzed:
                    st.markdown("<div class='ai-header'>🤖 AI 深度投資建議 (Investment Insight)</div>", unsafe_allow_html=True)
                    st.markdown(st.session_state['analysis_results'][stock_name])
                    
                st.markdown("</div>", unsafe_allow_html=True) 

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側選擇掃描策略，點擊 **「啟動全自動掃描」** 開始量化分析。")
