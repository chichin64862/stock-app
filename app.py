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
from datetime import datetime

# --- PDF 生成庫檢查 ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
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

# --- 2. CSS 針對性修復 (針對下拉選單可讀性優化) ---
st.markdown("""
<style>
    /* 1. 全局基底 */
    .stApp { background-color: #0e1117 !important; }
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #e6e6e6 !important;
        font-family: 'Roboto', sans-serif;
    }

    /* 2. DataFrame 右上角配置選單 (白底黑字) */
    div[role="menu"] div, div[role="menu"] span, div[role="menu"] label {
        color: #31333F !important;
        font-weight: 500 !important;
    }
    div[role="menu"] label { color: #31333F !important; }

    /* 3. 【關鍵修正】下拉選單 (解決白底灰字看不見的問題) */
    
    /* (A) 選單輸入框本體：維持深色，與側邊欄融合 */
    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        border-color: #4b4b4b !important;
        color: white !important;
    }
    
    /* (B) 彈出列表容器：強制設為【白色背景】，配合您的視覺現況 */
    div[data-baseweb="popover"], ul[data-baseweb="menu"] {
        background-color: #ffffff !important; 
        border: 1px solid #cccccc !important;
    }
    
    /* (C) 選項文字：強制設為【純黑色】，確保在白底上清晰可見 */
    div[data-baseweb="popover"] li, 
    div[data-baseweb="popover"] div, 
    li[role="option"] {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* (D) 滑鼠懸停與選中狀態：綠底白字 */
    li[role="option"]:hover, 
    li[role="option"][aria-selected="true"] {
        background-color: #238636 !important; /* 綠色高亮 */
        color: #ffffff !important; /* 白字 */
    }

    /* 4. 下載按鈕 */
    .stDownloadButton button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #238636 !important;
        white-space: nowrap !important;
        min-width: 180px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stDownloadButton button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }
    .stDownloadButton p { color: inherit !important; font-size: 1rem !important; }

    /* 5. Toolbar (強制深色) */
    [data-testid="stElementToolbar"] {
        background-color: #262730 !important;
        border: 1px solid #4b4b4b !important;
    }
    [data-testid="stElementToolbar"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stElementToolbar"] button:hover {
        background-color: #4b4b4b !important;
    }

    /* 6. 其他元件 */
    input { color: #ffffff !important; caret-color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stock-card {
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #30363d; 
        margin-bottom: 15px;
    }
    .pdf-center {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #238636;
        margin-bottom: 20px;
    }
    .ai-header { color: #58a6ff !important; font-weight: bold; font-size: 1.3rem; margin-bottom: 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'df_norm' not in st.session_state: st.session_state['df_norm'] = None

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

# --- 7. PDF 生成引擎 ---
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
    
    # 標題更新
    story.append(Paragraph(f"熵值決策選股及AI深度分析報告", title_style))
    story.append(Paragraph(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')} (僅供參考使用)", normal_style))
    story.append(Spacer(1, 20))

    for idx, stock in enumerate(stock_data_list):
        if idx > 0: story.append(PageBreak()) 
        name = stock['name']
        story.append(Paragraph(f"🎯 {name}", h2_style))
        story.append(Paragraph("_" * 60, normal_style))
        story.append(Spacer(1, 10))
        
        # 加入「戰略指令」
        action = stock.get('action', 'N/A')
        story.append(Paragraph(f"⚡ 系統戰略指令: <b>{action}</b>", h3_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("📊 核心數據概覽 (Key Metrics)", h3_style))
        t_data = [
            ["指標", "數值", "指標", "數值"],
            [f"收盤價", f"{stock['price']}", f"Entropy Score", f"{stock['score']}"],
            [f"PEG Ratio", f"{stock.get('peg', 'N/A')}", f"季線乖離", f"{stock.get('ma_bias', 'N/A')}"],
            [f"負債權益比", f"{stock.get('debt_eq', 'N/A')}", f"FCF Yield (現金流)", f"{stock.get('fcf_yield', 'N/A')}"],
            [f"合約負債", f"{stock.get('cl_val', '尚未讀取')}", f"Beta", f"{stock.get('beta', 'N/A')}"],
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
        if radar:
            story.append(Paragraph("⚡ 四大因子貢獻度", h3_style))
            best_factor = max(radar, key=radar.get)
            story.append(Paragraph(f"🚀 主力優勢: <b>{best_factor} ({radar[best_factor]:.1f}%)</b>", normal_style))
            r_data = [[k, f"{v:.1f}%"] for k, v in radar.items()]
            r_table = Table([["因子面向", "得分 (0-100)"]] + r_data, colWidths=[200, 100], hAlign='LEFT')
            r_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#16A085")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(r_table)
            story.append(Spacer(1, 15))

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

# --- 8. Gemini API ---
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

# Prompt
HEDGE_FUND_PROMPT = """
【指令】
請針對 **[STOCK]** 撰寫一份客觀的「投資決策分析報告」。

【⚠️ 分析邏輯指令】
請直接根據量化數據與產業現況進行分析，無需扮演任何角色或提及任何機構名稱。報告內容應包含：

1. **財務健康度評估**：
   - 結合「負債權益比」與「自由現金流」判斷公司體質與獲利含金量。
   - 評估是否有高槓桿或虛胖風險。

2. **營收與成長動能**：
   - 根據「合約負債」的金額與變動，**推算** 未來 1-2 季的訂單能見度。
   - 指出目前是處於「訂單滿載」、「庫存調整」還是「需求疲軟」階段。

3. **操作建議與風險提示**：
   - **投資評等**：請給出 [強力買進 / 區間操作 / 減持觀望] 建議。
   - **關鍵點位**：設定合理的「防禦區間 (Support)」與「目標區間 (Target)」。
   - **觀察指標**：列出未來最需要關注的一個風險變數。

【最新市場即時數據】
[DATA_CONTEXT]
"""

# --- 9. 數據處理 ---
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

# 指標配置
indicators_config = {
    'Price vs MA60': {'col': 'priceToMA60', 'direction': '負向', 'name': '季線乖離', 'category': '技術'},
    'Volume Change': {'col': 'volumeRatio', 'direction': '正向', 'name': '量能比', 'category': '籌碼'},
    'PEG Ratio': {'col': 'pegRatio', 'direction': '負向', 'name': 'PEG', 'category': '估值'},
    'Price To Book': {'col': 'priceToBook', 'direction': '負向', 'name': 'PB比', 'category': '估值'},
    'ROE': {'col': 'returnOnEquity', 'direction': '正向', 'name': 'ROE', 'category': '財報'},
    'Debt To Equity': {'col': 'debtToEquity', 'direction': '負向', 'name': '負債權益比', 'category': '財報'},
    'FCF Yield': {'col': 'fcfYield', 'direction': '正向', 'name': 'FCF收益率', 'category': '財報'},
}

def fetch_single_stock(ticker):
    try:
        parts = ticker.split(' ')
        symbol = parts[0]
        name_zh = parts[1] if len(parts) > 1 else symbol
        
        display_code = symbol.split('.')[0]
        stock = yf.Ticker(symbol)
        info = stock.info 
        name_en = info.get('shortName', '')
        final_name = f"{name_zh} ({name_en})" if name_en else name_zh

        peg = info.get('pegRatio', None)
        pe = info.get('trailingPE', None)
        growth = info.get('revenueGrowth', 0) 
        if peg is None and pe is not None and growth > 0: peg = pe / (growth * 100)
        elif peg is None: peg = 2.5 
        
        price = info.get('currentPrice', info.get('previousClose', 0))
        ma50 = info.get('fiftyDayAverage', price) 
        bias = (price / ma50) - 1 if ma50 and ma50 > 0 else 0
        
        vol_avg = info.get('averageVolume', 0)
        vol_curr = info.get('volume', 0)
        if vol_curr == 0 or vol_avg == 0:
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    vol_curr = hist['Volume'].iloc[-1]
                    vol_avg = hist['Volume'].mean()
            except: pass
        vol_ratio = (vol_curr / vol_avg) if vol_avg > 0 else 1.0
        
        fcf = info.get('freeCashflow', 0)
        if fcf is None: fcf = 0
        mkt_cap = info.get('marketCap', 1)
        if mkt_cap is None: mkt_cap = 1
        fcf_yield = (fcf / mkt_cap) if mkt_cap > 0 else 0
        
        return {
            '代號': display_code,
            'full_symbol': symbol,
            '名稱': final_name,
            'close_price': price, 
            'pegRatio': peg, 
            'priceToMA60': bias, 
            'volumeRatio': vol_ratio,
            'priceToBook': info.get('priceToBook', np.nan),
            'returnOnEquity': info.get('returnOnEquity', np.nan), 
            'debtToEquity': info.get('debtToEquity', np.nan),
            'fcfYield': fcf_yield * 100, 
            'beta': info.get('beta', 1.0)
        }
    except: return None

def get_stock_data_concurrent(selected_list):
    data = []
    progress_bar = st.progress(0, text="初始化平台資料庫...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(fetch_single_stock, t): t for t in selected_list}
        completed = 0
        total = len(selected_list)
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result: data.append(result)
            completed += 1
            progress_bar.progress(completed / total, text=f"正在掃描市場數據: {completed}/{total}...")
    progress_bar.empty()
    return pd.DataFrame(data)

def calculate_entropy_score(df, config):
    df = df.dropna().copy()
    if df.empty: return df, None, "No valid data found.", None
    
    if 'returnOnEquity' in df.columns:
        df = df[df['returnOnEquity'] > 0]
        
    if df.empty: return df, None, "所有股票皆未通過剛性過濾 (ROE > 0)", None

    df_norm = df.copy()
    
    for key, cfg in config.items():
        col = cfg['col']
        if col in df.columns:
            q_low = df[col].quantile(0.05)
            q_high = df[col].quantile(0.95)
            df_norm[col] = df[col].clip(lower=q_low, upper=q_high)
            
            mn, mx = df_norm[col].min(), df_norm[col].max()
            denom = mx - mn
            if denom == 0: df_norm[f'{col}_n'] = 0.5
            else:
                if cfg['direction'] == '正向': df_norm[f'{col}_n'] = (df_norm[col] - mn) / denom
                else: df_norm[f'{col}_n'] = (mx - df_norm[col]) / denom
            
    m = len(df)
    k = 1 / np.log(m) if m > 1 else 0
    weights = {}
    for key, cfg in config.items():
        col = cfg['col']
        if col in df_norm.columns and f'{col}_n' in df_norm.columns:
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

def get_contract_liabilities_safe(symbol_code):
    try:
        if not symbol_code.endswith('.TW') and not symbol_code.endswith('.TWO'): symbol_code += '.TW'
        stock = yf.Ticker(symbol_code)
        bs = stock.balance_sheet
        if bs.empty: return "無財報數據"
        target_keys = ['Contract Liabilities', 'Deferred Revenue']
        val = None
        for key in target_keys:
            matches = [k for k in bs.index if key in k]
            if matches:
                val = bs.loc[matches[0]].iloc[0]
                break
        if val is not None and not pd.isna(val): return f"{val / 100000000:.2f} 億元"
        else: return "無合約負債數據"
    except: return "讀取失敗"

def get_radar_data(df_norm_row, config):
    categories = {'技術': [], '籌碼': [], '財報': [], '估值': []}
    for key, cfg in config.items():
        cat = cfg['category']
        col_n = f"{cfg['col']}_n"
        if col_n in df_norm_row:
            score = df_norm_row[col_n] * 100
            categories[cat].append(score)
    return {k: np.mean(v) if v else 0 for k, v in categories.items()}

def plot_radar_chart(row_name, radar_data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(radar_data.values()), theta=list(radar_data.keys()),
        fill='toself', name=row_name, line_color='#00e676', fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='#8b949e'), bgcolor='rgba(0,0,0,0)'),
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e6e6e6', size=12), height=250
    )
    return fig

def render_factor_bars(radar_data):
    html = ""
    colors = {'技術': '#29b6f6', '籌碼': '#ab47bc', '財報': '#ffca28', '估值': '#ef5350'}
    for cat, score in radar_data.items():
        color = colors.get(cat, '#8b949e')
        blocks = int(score / 10)
        visual_bar = "■" * blocks + "░" * (10 - blocks)
        html += f"""<div style="margin-bottom: 8px;"><div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#e6e6e6;"><span><span style="color:{color};">●</span> {cat}</span><span>{score:.0f}%</span></div><div style="font-family: monospace; color:{color}; letter-spacing: 2px;">{visual_bar}</div></div>"""
    return html

# --- 11. 側邊欄與執行 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.markdown("---")
    scan_mode = st.radio("選股模式：", ["🔥 熱門策略掃描", "🏭 產業類股掃描", "自行輸入/多選"], label_visibility="collapsed")
    target_stocks = []
    
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
            
    st.info(f"已鎖定 {len(target_stocks)} 檔標的")
    st.markdown("---")
    run_btn = st.button("🚀 啟動全自動掃描", type="primary", use_container_width=True)

# --- 12. 主儀表板 ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股及AI深度分析平台")
    st.caption("Entropy Scoring • Factor Radar • PDF Reporting (僅供參考使用)")
with col2:
    if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
         st.metric("Total Scanned", f"{len(st.session_state['raw_data'])} Stocks", delta="Live Update")

if run_btn:
    if not target_stocks:
        st.warning("⚠️ Please select at least one stock or strategy from the sidebar.")
    else:
        st.session_state['analysis_results'] = {}
        st.session_state['raw_data'] = None
        st.session_state['df_norm'] = None
        raw = get_stock_data_concurrent(target_stocks)
        if not raw.empty:
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    # 檢測資料完整性
    required_cols = ['fcfYield', 'debtToEquity']
    if not all(col in st.session_state['raw_data'].columns for col in required_cols):
        st.toast("⚠️ 偵測到系統升級，正在重新抓取最新財報數據...", icon="🔄")
        st.session_state['raw_data'] = None
        st.rerun()

    raw = st.session_state['raw_data']
    res, w, err, df_norm = calculate_entropy_score(raw, indicators_config)
    st.session_state['df_norm'] = df_norm 
    
    # 增加趨勢判定欄位 (Trend)
    def get_trend_label(bias):
        if bias < -0.05: return "🟢 超跌/買點"
        elif bias > 0.15: return "🔴 過熱/賣點"
        else: return "🟡 盤整/持有"
        
    # 增加戰略指令 (Action Plan)
    def determine_action_plan(row):
        score = row['Score']
        bias = row['priceToMA60']
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
    
    if err:
        st.error(err)
    else:
        res['Trend'] = res['priceToMA60'].apply(get_trend_label)
        res['Action Plan'] = res.apply(determine_action_plan, axis=1)
        top_n = 10
        top_stocks = res.head(top_n)

        st.markdown("### 🏆 Top 10 潛力標的 (Entropy Ranking)")
        st.dataframe(
            top_stocks[['代號', '名稱', 'close_price', 'Score', 'pegRatio', 'priceToMA60', 'debtToEquity', 'fcfYield', 'Action Plan']],
            column_config={
                "Score": st.column_config.ProgressColumn("Entropy Score", format="%.1f", min_value=0, max_value=100),
                "close_price": st.column_config.NumberColumn("Price", format="%.2f"),
                "pegRatio": st.column_config.NumberColumn("PEG", format="%.2f"),
                "priceToMA60": st.column_config.NumberColumn("MA Bias", format="%.2%"),
                "debtToEquity": st.column_config.NumberColumn("D/E (Risk)", format="%.2f"),
                "fcfYield": st.column_config.NumberColumn("FCF Yield", format="%.2f%%"),
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
                                'peg': row['pegRatio'],
                                'beta': row.get('beta', 0),
                                'debt_eq': row.get('debtToEquity', 'N/A'),
                                'fcf_yield': f"{row.get('fcfYield', 0):.2f}%",
                                'ma_bias': f"{row['priceToMA60']:.2%}",
                                'radar_data': radar,
                                'analysis': analysis_text,
                                'action': row['Action Plan']
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
                        fig_radar = plot_radar_chart(row['名稱'], radar_data)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with c2:
                        st.markdown("**因子貢獻解析**")
                        st.markdown(render_factor_bars(radar_data), unsafe_allow_html=True)
                
                with c3:
                    st.markdown("**配置時機判定 (Trend vs Value)**")
                    ticker_for_chart = row['full_symbol']
                    try:
                        stock_hist = yf.Ticker(ticker_for_chart).history(period="6mo")
                        if not stock_hist.empty:
                            fig_trend = go.Figure()
                            fig_trend.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['Close'], mode='lines', name='Price', line=dict(color='#29b6f6', width=2)))
                            last_price = stock_hist['Close'].iloc[-1]
                            fig_trend.add_trace(go.Scatter(x=[stock_hist.index[-1]], y=[last_price], mode='markers', marker=dict(color='#00e676', size=10), name='Current'))
                            
                            timing_msg = "🟢 最佳佈局點 (Value Zone)" if row['priceToMA60'] < 0 else "🟡 持有/觀察 (Momentum)"
                            if row['priceToMA60'] > 0.15: timing_msg = "🔴 留意過熱 (Overheated)"
                            
                            fig_trend.update_layout(
                                title=dict(text=timing_msg, font=dict(size=14, color='#e6e6e6')),
                                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30363d'),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=0,r=0,t=30,b=0), height=250, showlegend=False
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else: st.warning("⚠️ 無法取得歷史數據")
                    except Exception as e: st.error("圖表載入失敗")

                col_btn, col_dl = st.columns([3, 1])
                
                with col_btn:
                     if st.button(f"✨ 生成分析報告", key=f"btn_{i}", use_container_width=True, disabled=is_analyzed):
                         if not is_analyzed:
                            with st.spinner(f"⚡ AI 正在為您撰寫 {stock_name} 的投資備忘錄..."):
                                cl_val = get_contract_liabilities_safe(row['full_symbol']) 
                                fcf_val = row.get('fcfYield', 0)
                                de_val = row.get('debtToEquity', 0)
                                real_time_data = f"""
                                - 收盤價: {row['close_price']}
                                - 合約負債: {cl_val}
                                - 自由現金流收益率 (FCF Yield): {fcf_val:.2f}%
                                - 負債權益比 (D/E): {de_val:.2f}
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
                        'peg': row['pegRatio'],
                        'debt_eq': row.get('debtToEquity', 'N/A'),
                        'fcf_yield': f"{row.get('fcfYield', 0):.2f}%",
                        'ma_bias': f"{row['priceToMA60']:.2%}",
                        'radar_data': radar_data,
                        'analysis': st.session_state['analysis_results'].get(stock_name, None),
                        'action': row['Action Plan']
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
