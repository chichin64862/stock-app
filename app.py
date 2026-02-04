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
    page_title="熵值決策選股及AI深度分析平台 (Buffett Edition)", 
    page_icon="🎩", 
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
    [data-testid="stExpander"] { background-color: #262730 !important; border: 1px solid #4b4b4b !important; border-radius: 5px; }
    .buffett-box { padding: 10px; background-color: rgba(255, 215, 0, 0.1); border: 1px solid #FFD700; border-radius: 5px; color: #FFD700; margin-bottom: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'df_norm' not in st.session_state: st.session_state['df_norm'] = None
if 'market_fundamentals' not in st.session_state: st.session_state['market_fundamentals'] = {}
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None
if 'av_api_key' not in st.session_state: st.session_state['av_api_key'] = "59P38LL8MKU2XB1M"

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
        
        # 顯示巴菲特標籤
        if stock.get('is_buffett', False):
            story.append(Paragraph(f"🎩 <b>Buffett Pick (巴菲特選股)</b>", h3_style))
        
        action = stock.get('action', 'N/A')
        story.append(Paragraph(f"⚡ 系統戰略指令: <b>{action}</b>", h3_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("📊 核心數據概覽 (Key Metrics)", h3_style))
        
        pe_val = stock.get('pe', 'N/A')
        pb_val = stock.get('pb', 'N/A')
        yield_val = stock.get('yield', 'N/A')
        peg_val = stock.get('syn_peg', 'N/A')
        
        t_data = [
            ["指標", "數值", "指標", "數值"],
            [f"收盤價", f"{stock['price']}", f"Entropy Score", f"{stock['score']}"],
            [f"本益比 (P/E)", f"{pe_val}", f"季線乖離", f"{stock.get('ma_bias', 'N/A')}"],
            [f"股價淨值比 (P/B)", f"{pb_val}", f"殖利率 (Yield)", f"{yield_val}%"],
            [f"合成 PEG", f"{peg_val}", f"營收成長 (YoY)", f"{stock.get('rev_growth', 'N/A')}%"],
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

【⚠️ 成長爆發力分析指令】
請扮演華爾街成長股分析師，專注於尋找「半年內可能爆發」的標的：

1. **成長動能檢測 (Momentum)**：
   - 檢視 **營收成長率 (Revenue YoY)**：是否有加速趨勢？(若 > 20% 為高成長)。
   - 檢視 **合成 PEG**：是否在合理範圍 (< 1.5)？這代表成長撐得起估值。

2. **籌碼與風險 (Chips & Risk)**：
   - 檢視 **法人買賣超**：是否有機構法人進駐佈局？
   - 下檔保護：殖利率是否足夠？

3. **操作建議**：
   - **投資評等**：[強力買進 / 區間操作 / 減持觀望]。
   - **目標**：半年內是否具備 20% 以上潛在漲幅？
   - **催化劑**：未來一季最重要的觀察重點是什麼？

【最新市場即時數據】
[DATA_CONTEXT]
"""

# --- 10. 數據處理 ---
@st.cache_data
def get_tw_stock_info():
    codes = twstock.codes
    stock_dict = {} 
    industry_dict = {} 
    # 建立產業映射
    industry_map = {}
    
    for code, info in codes.items():
        if info.type == '股票':
            if info.market == '上市': suffix = '.TW'
            elif info.market == '上櫃': suffix = '.TWO'
            else: continue
            full_code = f"{code}{suffix}"
            name = info.name
            industry = info.group # 產業類別
            stock_dict[full_code] = f"{full_code} {name}"
            if industry not in industry_dict: industry_dict[industry] = []
            industry_dict[industry].append(full_code)
            industry_map[code] = industry
            
    return stock_dict, industry_dict, industry_map

stock_map, industry_map, global_industry_lookup = get_tw_stock_info()

# --- 【核心】動態產業權重引擎 (Dynamic Sector Weighting) ---
def get_sector_config(industry_name):
    # 基礎指標 (所有股票都要看)
    base_config = {
        'Price vs MA60': {'col': 'priceToMA60', 'direction': '負向', 'name': '季線乖離', 'category': '技術'},
        'Volume Change': {'col': 'volumeRatio', 'direction': '正向', 'name': '量能比', 'category': '籌碼'},
        'Volatility': {'col': 'volatility', 'direction': '負向', 'name': '波動率', 'category': '風險'}, 
    }
    
    # 定義產業原型
    # 1. 科技成長 (Growth)
    tech_sectors = ['半導體業', '電腦及週邊設備業', '電子零組件業', '光電業', '通信網路業', '電子通路業', '資訊服務業', '其他電子業']
    # 2. 金融 (Financial)
    fin_sectors = ['金融保險業']
    # 3. 傳產循環 (Cyclical)
    cyclical_sectors = ['水泥工業', '食品工業', '塑膠工業', '紡織纖維', '電機機械', '電器電纜', '玻璃陶瓷', '造紙工業', '鋼鐵工業', '橡膠工業', '汽車工業', '建材營造', '航運業', '油電燃氣業']
    
    # 動態配置
    if industry_name in tech_sectors:
        # 科技股：看成長 (PEG, YoY)
        base_config.update({
            'Revenue Growth': {'col': 'rev_growth', 'direction': '正向', 'name': '營收成長', 'category': '成長'},
            'Synthetic PEG': {'col': 'syn_peg', 'direction': '負向', 'name': '合成 PEG', 'category': '成長'},
            'Earnings Yield': {'col': 'ep_ratio', 'direction': '正向', 'name': '獲利收益率', 'category': '估值'},
        })
    elif industry_name in fin_sectors:
        # 金融股：看配息 (Yield) 與 淨值 (PB)
        base_config.update({
            'Dividend Yield': {'col': 'yield', 'direction': '正向', 'name': '殖利率', 'category': '財報'},
            'P/B Ratio': {'col': 'pb', 'direction': '負向', 'name': '淨值比', 'category': '估值'},
            'Synthetic ROE': {'col': 'roe_syn', 'direction': '正向', 'name': '合成ROE', 'category': '財報'},
        })
    elif industry_name in cyclical_sectors:
        # 傳產股：看本益比 (PE) 與 籌碼
        base_config.update({
            'P/E Ratio': {'col': 'pe', 'direction': '負向', 'name': '本益比', 'category': '估值'},
            'Institutions Buy': {'col': 'chips', 'direction': '正向', 'name': '法人買賣超', 'category': '籌碼'},
            'P/B Ratio': {'col': 'pb', 'direction': '負向', 'name': '淨值比', 'category': '估值'},
        })
    else:
        # 其他：通用配置
        base_config.update({
            'P/E Ratio': {'col': 'pe', 'direction': '負向', 'name': '本益比', 'category': '估值'},
            'Dividend Yield': {'col': 'yield', 'direction': '正向', 'name': '殖利率', 'category': '財報'},
            'Revenue Growth': {'col': 'rev_growth', 'direction': '正向', 'name': '營收成長', 'category': '成長'},
        })
        
    return base_config

# --- Alpha Vantage 核心 ---
def fetch_alpha_vantage_data(symbol, api_key):
    if not api_key: return None
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): symbol += '.TW'
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
        r = requests.get(url, timeout=5)
        data = r.json()
        url_price = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        r_price = requests.get(url_price, timeout=5)
        data_price = r_price.json()
        
        if "Global Quote" in data_price and "05. price" in data_price["Global Quote"]:
            price = float(data_price["Global Quote"]["05. price"])
            pe = float(data.get("PERatio", 0)) if data.get("PERatio") and data.get("PERatio") != "None" else 0
            pb = float(data.get("PriceToBookRatio", 0)) if data.get("PriceToBookRatio") and data.get("PriceToBookRatio") != "None" else 0
            dy = float(data.get("DividendYield", 0)) * 100 if data.get("DividendYield") and data.get("DividendYield") != "None" else 0
            peg = float(data.get("PEGRatio", 0)) if data.get("PEGRatio") and data.get("PEGRatio") != "None" else 0
            return {'price': price, 'pe': pe, 'pb': pb, 'yield': dy, 'peg': peg, 'source': 'AV'}
    except: return None
    return None

def safe_float(val):
    try:
        if isinstance(val, (int, float)): return float(val)
        val = str(val).replace(',', '').strip()
        if val == '-' or val == '': return 0.0
        return float(val)
    except: return 0.0

# --- TWSE 數據連接器 (含爬蟲) ---
@st.cache_data(ttl=3600)
def fetch_market_data_advanced():
    market_data = {}
    
    # 1. 本益比 (使用 pd.read_html 爬蟲)
    try:
        url = "https://www.twse.com.tw/exchangeReport/BWIBBU_ALL?response=html"
        dfs = pd.read_html(url)
        if dfs:
            df = dfs[0]
            for _, row in df.iterrows():
                try:
                    code = str(row[0]).split(' ')[0]
                    market_data[code] = {
                        'pe': safe_float(row[4]), 'pb': safe_float(row[5]), 'yield': safe_float(row[2])
                    }
                except: pass
    except: pass
    
    # 2. 三大法人 (使用 pd.read_html)
    try:
        url = "https://www.twse.com.tw/fund/T86?response=html&selectType=ALL"
        dfs = pd.read_html(url)
        if dfs:
            df = dfs[0]
            for _, row in df.iterrows():
                try:
                    code = str(row[0]).split(' ')[0]
                    if code in market_data:
                        # 倒數第二欄是三大法人合計
                        market_data[code]['chips'] = safe_float(row.iloc[-2]) / 1000
                except: pass
    except: pass
    
    # 3. 營收成長 (使用 Open Data API，較穩定)
    try:
        url = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
        r = requests.get(url, verify=False, timeout=10) # 忽略 SSL 錯誤
        if r.status_code == 200:
            for item in r.json():
                code = item.get('公司代號')
                if code in market_data:
                    rev = item.get('營業收入-去年同月增減百分比', 0)
                    market_data[code]['rev_growth'] = safe_float(rev)
    except: pass
            
    return market_data

def get_radar_data(df_norm_row, config):
    categories = {'技術': [], '籌碼': [], '財報': [], '估值': [], '成長': [], '風險': []}
    for key, cfg in config.items():
        cat = cfg['category']
        col_n = f"{cfg['col']}_n"
        if col_n in df_norm_row:
            score = df_norm_row[col_n] * 100
            categories[cat].append(score)
    # 過濾空值
    return {k: np.mean(v) if v else 0 for k, v in categories.items() if v}

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

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hybrid_data(tickers_list, tej_data=None, use_av=False, av_key=None):
    results = []
    
    if use_av and av_key:
        # AV 模式 (略，同前)
        pass 

    fund_map = fetch_market_data_advanced()
    
    try:
        symbols = [t.split(' ')[0] for t in tickers_list]
        data = yf.download(symbols, period="6mo", group_by='ticker', progress=False, threads=False)
        
        for ticker_full in tickers_list:
            parts = ticker_full.split(' ')
            symbol = parts[0]
            name = parts[1] if len(parts) > 1 else symbol
            code = symbol.split('.')[0]
            
            # 取得產業別
            industry = global_industry_lookup.get(code, 'Other')
            
            price = np.nan; ma_bias = 0; vol_ratio = 1.0; volatility = 0.05
            
            try:
                df = data if len(symbols) == 1 else (data[symbol] if symbol in data else pd.DataFrame())
                if not df.empty and 'Close' in df.columns:
                    df = df.dropna(subset=['Close'])
                    if not df.empty:
                        latest = df.iloc[-1]
                        price = float(latest['Close'])
                        if not pd.isna(price):
                            ma60 = df['Close'].rolling(window=60).mean().iloc[-1]
                            if not pd.isna(ma60) and ma60 > 0: ma_bias = (price / ma60) - 1
                            vol_curr = df['Volume'].iloc[-1]
                            vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
                            if not pd.isna(vol_avg) and vol_avg > 0: vol_ratio = vol_curr / vol_avg
                            volatility = df['Close'].pct_change().std() * (252 ** 0.5)
            except: pass
            
            if pd.isna(price):
                try:
                    realtime = twstock.realtime.get(code)
                    if realtime['success']:
                        p_str = realtime['realtime'].get('latest_trade_price', '-')
                        if p_str == '-' or not p_str: p_str = realtime['realtime'].get('best_bid_price', [None])[0]
                        if p_str and p_str != '-': price = float(p_str)
                except: pass
            
            if not pd.isna(price):
                f_data = fund_map.get(code, {'pe': 0, 'pb': 0, 'yield': 0, 'rev_growth': 0, 'chips': 0})
                pe = f_data.get('pe', 0)
                pb = f_data.get('pb', 0)
                dy = f_data.get('yield', 0)
                rev_growth = f_data.get('rev_growth', 0)
                chips = f_data.get('chips', 0)
                is_tej = False
                source = 'TWSE'
                
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    # TEJ 覆蓋邏輯 (略)
                    is_tej = True
                    source = 'TEJ'

                roe_syn = 0
                if pe > 0 and pb > 0: roe_syn = (pb / pe) * 100
                elif pe == 0: roe_syn = -5.0
                if pd.isna(volatility): volatility = 0.5
                
                ep_ratio = (1/pe * 100) if pe > 0 else 0
                
                syn_peg = 100.0 # 預設高值 (不吸引人)
                if pe > 0 and rev_growth > 0:
                    syn_peg = pe / rev_growth
                
                # 【新增】巴菲特邏輯判斷
                is_buffett = False
                if roe_syn > 15 and volatility < 0.3 and (syn_peg < 1.5 or pe < 20):
                    is_buffett = True
                
                results.append({
                    '代號': code, 'full_symbol': symbol, '名稱': name, 'close_price': price,
                    'priceToMA60': ma_bias, 'volumeRatio': vol_ratio, 'volatility': volatility,
                    'pe': pe, 'pb': pb, 'yield': dy, 'roe_syn': roe_syn, 'beta': 1.0,
                    'rev_growth': rev_growth, 'chips': chips, 'syn_peg': syn_peg,
                    'ep_ratio': ep_ratio,
                    'is_tej': is_tej, 'source': source, 'industry': industry,
                    'is_buffett': is_buffett,
                    'pegRatio': np.nan, 'debtToEquity': np.nan, 'fcfYield': np.nan
                })
    except Exception: pass
            
    return pd.DataFrame(results)

def calculate_entropy_score(df, use_buffett=False):
    if df.empty: return df, None, "數據抓取為空。", None
    df_norm = df.copy()
    
    # 遍歷每一列，根據該股票的產業選擇指標權重
    scores = []
    
    for idx, row in df.iterrows():
        # 1. 獲取該股票的產業配置
        config = get_sector_config(row['industry'])
        
        stock_score = 0
        total_weight = 0
        
        # 2. 計算該股票的得分 (簡化版熵值：直接正規化加權)
        # 因為不同產業指標不同，無法用全矩陣熵值法，改用「相對排名法」
        for key, cfg in config.items():
            col = cfg['col']
            val = row.get(col, 0)
            
            # 取得全體該欄位的分位數 (由全體數據決定相對位置)
            all_vals = df[col].fillna(0)
            
            if cfg['direction'] == '正向':
                rank = all_vals.rank(pct=True)[idx] # 百分位 0~1
            else:
                rank = 1 - all_vals.rank(pct=True)[idx]
            
            # 給予權重 (這裡假設均權，或可微調)
            weight = 1.0
            stock_score += rank * weight
            total_weight += weight
            
            # 儲存正規化值以便畫雷達圖
            df_norm.loc[idx, f'{col}_n'] = rank
            
        final_score = (stock_score / total_weight) * 100
        
        # 【核心】巴菲特加分
        if use_buffett and row['is_buffett']:
            final_score += 10 # 獎勵分
            if final_score > 100: final_score = 100
            
        scores.append(final_score)
        
    df['Score'] = scores
    df['Score'] = df['Score'].round(1)
    
    return df.sort_values('Score', ascending=False), None, None, df_norm

def render_factor_bars(radar_data):
    html = ""
    colors = {'技術': '#29b6f6', '籌碼': '#ab47bc', '財報': '#ffca28', '估值': '#ef5350', '成長': '#00e676'}
    for cat, score in radar_data.items():
        color = colors.get(cat, '#8b949e')
        blocks = int(score / 10)
        visual_bar = "■" * blocks + "░" * (10 - blocks)
        html += f"""<div style="margin-bottom: 8px;"><div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#e6e6e6;"><span><span style="color:{color};">●</span> {cat}</span><span>{score:.0f}%</span></div><div style="font-family: monospace; color:{color}; letter-spacing: 2px;">{visual_bar}</div></div>"""
    return html

# --- 12. 主儀表板與流程 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.markdown("### 1️⃣ 數據源設定")
    
    with st.expander("🔑 Alpha Vantage 設定 (AV Key)", expanded=True):
        av_key = st.text_input("API Key", value=st.session_state.get('av_api_key', ''))
        if av_key: st.session_state['av_api_key'] = av_key
        st.caption("✅ 已掛載：精準模式與分析報告將使用 AV 數據")

    with st.expander("📂 匯入 TEJ 數據 (選填)", expanded=False):
        uploaded_file = st.file_uploader("上傳 Excel/CSV", type=['csv', 'xlsx'])
        if uploaded_file:
            tej_data = process_tej_upload(uploaded_file)
            if tej_data: 
                st.session_state['tej_data'] = tej_data
                st.markdown(f"<div class='success-box'>✅ TEJ 數據已就緒：{len(tej_data)} 檔</div>", unsafe_allow_html=True)

    fund_map = st.session_state.get('market_fundamentals', {})
    if len(fund_map) > 0:
        st.success(f"📊 官方大數據：已載入 {len(fund_map)} 檔 (含營收、籌碼)")
    else:
        st.warning("⚠️ 數據未載入 (請按下方重置)")

    if st.button("🔴 清除快取並重置", use_container_width=True):
        st.cache_data.clear()
        if 'raw_data' in st.session_state: del st.session_state['raw_data']
        if 'scan_finished' in st.session_state: del st.session_state['scan_finished']
        if 'market_fundamentals' in st.session_state: del st.session_state['market_fundamentals']
        st.rerun()
        
    st.markdown("---")
    
    # 2. 選股策略
    st.markdown("### 2️⃣ 選股策略")
    
    # 巴菲特模式開關
    use_buffett = st.checkbox("🎩 啟用巴菲特選股邏輯 (價值+護城河)", value=False)
    if use_buffett:
        st.markdown("<div class='buffett-box'>💡 已啟用：將針對高 ROE、低波動、合理估值之標的給予加權。</div>", unsafe_allow_html=True)
    
    scan_mode = st.radio("選股模式：", ["🔥 熱門策略掃描", "🏭 產業類股掃描", "自行輸入/多選"], label_visibility="collapsed")
    target_stocks = []
    
    if scan_mode == "自行輸入/多選":
        st.caption("🔍 若找不到股票，請直接輸入代號:")
        manual_input = st.text_input("手動輸入代號:", placeholder="例如: 1802 或 2330", label_visibility="collapsed")
        default_selection = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海"]
        selected = st.multiselect("選擇股票:", options=sorted(list(stock_map.values())), default=[s for s in default_selection if s in stock_map.values()])
        target_stocks = selected
        if manual_input: target_stocks.append(manual_input)
        
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
            
    st.info(f"已鎖定 {len(target_stocks)} 檔標的")
    st.markdown("---")
    run_btn = st.button("🚀 啟動全自動掃描", type="primary", use_container_width=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股及AI深度分析平台 (Buffett Edition)")
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
        
        with st.spinner(f"🚀 正在啟動混合掃描 (動態產業權重 + 巴菲特邏輯)..."):
            raw = fetch_hybrid_data(
                target_stocks, 
                st.session_state.get('tej_data'),
                use_av=False,
                av_key=st.session_state.get('av_api_key')
            )
            
        if not raw.empty:
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()
        else:
            st.error("❌ 掃描失敗：無法獲取數據，請檢查 API Key 或網路連線。")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    if 'pe' not in st.session_state['raw_data'].columns:
        st.session_state['raw_data'] = None
        st.rerun()

    raw = st.session_state['raw_data']
    res, w, err, df_norm = calculate_entropy_score(raw, use_buffett)
    
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
            rev_growth = row.get('rev_growth', 0)
            syn_peg = row.get('syn_peg', 100)
            
            # 爆發型條件：高分 + 高成長
            if score >= 70 and rev_growth > 20:
                if syn_peg < 1.0: return "🚀 超級成長股 (Super Growth)"
                elif syn_peg < 1.5: return "🔥 動能強勢 (High Momentum)"
            
            if score >= 60: return "🟡 穩健持有 (Hold)"
            else: return "⛔ 觀望/賣出 (Avoid/Sell)"
        
        res['Trend'] = res['priceToMA60'].apply(get_trend_label)
        res['Action Plan'] = res.apply(determine_action_plan, axis=1)
        
        st.markdown("### 🏆 Top 10 潛力標的 (Entropy Ranking)")
        st.dataframe(
            res[['代號', '名稱', 'close_price', 'Score', 'pe', 'rev_growth', 'chips', 'syn_peg', 'Action Plan']],
            column_config={
                "Score": st.column_config.ProgressColumn("Entropy Score", format="%.1f", min_value=0, max_value=100),
                "close_price": st.column_config.NumberColumn("Price", format="%.2f"),
                "pe": st.column_config.NumberColumn("P/E", format="%.2f"),
                "rev_growth": st.column_config.NumberColumn("營收YoY", format="%.2f%%"),
                "chips": st.column_config.NumberColumn("法人買賣超(張)", format="%d"),
                "syn_peg": st.column_config.NumberColumn("合成PEG", format="%.2f"),
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
                            # 取得該股票對應的 radar config
                            config = get_sector_config(row.get('industry'))
                            radar = get_radar_data(norm_row, config)
                            analysis_text = st.session_state['analysis_results'].get(stock_name, None)
                            
                            bulk_data_final.append({
                                'name': stock_name,
                                'price': row['close_price'],
                                'score': row['Score'],
                                'pe': row.get('pe', 0),
                                'pb': row.get('pb', 0),
                                'fcf_yield': f"{row.get('yield', 0):.2f}%",
                                'rev_growth': f"{row.get('rev_growth', 0):.2f}%",
                                'chips': f"{row.get('chips', 0):.0f}",
                                'roe_syn': f"{row.get('roe_syn', 0):.2f}%",
                                'ma_bias': f"{row['priceToMA60']:.2%}",
                                'volatility': f"{row.get('volatility', 0):.2%}",
                                'radar_data': radar,
                                'analysis': analysis_text,
                                'action': row['Action Plan'],
                                'full_symbol': row['full_symbol'],
                                'source': row.get('source', ''),
                                'is_buffett': row.get('is_buffett', False)
                            })
                    
                    if bulk_data_final:
                        pdf_data_final = create_pdf(bulk_data_final)
                        st.download_button(
                            label="📑 下載個股 PDF",
                            data=pdf_data_final,
                            file_name=f"AlphaCore_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 深度戰略分析 (Strategic Deep Dive)")
        
        for i, (index, row) in enumerate(res.head(10).iterrows()):
            stock_name = f"{row['代號']} {row['名稱']}"
            is_analyzed = (stock_name in st.session_state['analysis_results'])
            
            with st.container():
                st.markdown(f"""<div class="stock-card"><h3>{stock_name} <span style="font-size:0.6em;color:#8b949e">NT$ {row['close_price']}</span> <span style="font-size:0.8em;color:#00e676;border:1px solid #00e676;padding:2px 5px;border-radius:4px;margin-left:10px;">{row['Action Plan']}</span></h3>""", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1.5, 1.2, 2])
                
                if index in df_norm.index:
                    norm_row = df_norm.loc[index]
                    config = get_sector_config(row.get('industry'))
                    radar_data = get_radar_data(norm_row, config)
                
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
                     btn_label = "✨ 生成分析報告 (AV 加強)" if st.session_state.get('av_api_key') else "✨ 生成分析報告"
                     if st.button(btn_label, key=f"btn_{i}", use_container_width=True, disabled=is_analyzed):
                         if not is_analyzed:
                            with st.spinner(f"⚡ AI 正在調用 Alpha Vantage 獲取 {stock_name} 精準數據..."):
                                av_data = None
                                if st.session_state.get('av_api_key'):
                                    av_data = fetch_alpha_vantage_data(row['full_symbol'], st.session_state['av_api_key'])
                                
                                pe_val = av_data['pe'] if av_data else row.get('pe', 0)
                                pb_val = av_data['pb'] if av_data else row.get('pb', 0)
                                dy_val = av_data['yield'] if av_data else row.get('yield', 0)
                                peg_val = av_data['peg'] if av_data else row.get('syn_peg', 0)
                                
                                real_time_data = f"""
                                - 收盤價: {row['close_price']}
                                - 本益比 (P/E): {pe_val:.2f}
                                - 股價淨值比 (P/B): {pb_val:.2f}
                                - 殖利率 (Yield): {dy_val:.2f}%
                                - 營收成長率 (YoY): {row.get('rev_growth', 0):.2f}%
                                - PEG Ratio: {peg_val:.2f}
                                - 法人買賣超: {row.get('chips', 0):.0f} 張
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
                        'rev_growth': f"{row.get('rev_growth', 0):.2f}%",
                        'chips': f"{row.get('chips', 0):.0f}",
                        'roe_syn': f"{row.get('roe_syn', 0):.2f}%",
                        'ma_bias': f"{row['priceToMA60']:.2%}",
                        'volatility': f"{row.get('volatility', 0):.2%}",
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
