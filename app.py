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
    page_title="熵值決策選股平台 (Final Visual Fix)", 
    page_icon="⚖️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 強制修復 (解決文字看不見問題) ---
st.markdown("""
<style>
    /* 強制全域深色主題，避免淺色模式導致白字消失 */
    .stApp { background-color: #0e1117 !important; }
    
    /* 強制側邊欄深色背景，確保與白字對比 */
    [data-testid="stSidebar"] { 
        background-color: #262730 !important; 
        border-right: 1px solid #4b4b4b;
    }
    
    /* 強制所有文字顏色為淺灰/白 */
    body, h1, h2, h3, h4, h5, h6, p, li, span, label { 
        color: #e6e6e6 !important; 
        font-family: 'Roboto', sans-serif; 
    }
    
    /* 修正輸入框與選單的可讀性 */
    div[data-baseweb="select"] > div, input { 
        background-color: #1f2937 !important; 
        color: white !important; 
        border: 1px solid #4b4b4b !important;
    }
    
    /* 個股卡片樣式 */
    .stock-card { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .buffett-tag { 
        background-color: #FFD700; color: #000; 
        padding: 3px 8px; border-radius: 12px; 
        font-weight: bold; font-size: 0.8rem; margin-left: 10px; 
    }
    .sector-tag {
        background-color: #2e7d32; color: #fff;
        padding: 3px 8px; border-radius: 12px;
        font-size: 0.8rem; margin-right: 5px;
    }
    
    /* 按鈕樣式 */
    .stButton button { background-color: #238636; color: white; border: none; }
    .stButton button:hover { background-color: #2ea043; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'tej_data' not in st.session_state: st.session_state['tej_data'] = None
if 'history_data' not in st.session_state: st.session_state['history_data'] = {}

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

# --- 6. 核心數據引擎 (Yahoo Only + Sanitization) ---
def get_stock_fundamentals(symbol):
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): symbol += '.TW'
        ticker = yf.Ticker(symbol)
        info = ticker.info 
        
        # 抓取並初步清洗
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
    """計算合成 PEG"""
    if pe and growth_rate and growth_rate > 0:
        # growth_rate 輸入為小數 (0.2)，計算需 *100 變 20
        return pe / (growth_rate * 100)
    return None

def sanitize_data(df):
    """
    【數據清洗核心】
    修正 Yahoo 單位不一致問題 (如 400% 殖利率)
    """
    if df.empty: return df
    
    # 1. 修正殖利率 (Yield)
    # 邏輯：如果大於 20%，通常是單位錯誤 (原本就是%又乘了100)，除以 100
    # 如果大於 1000%，強制設為 NaN
    if 'yield' in df.columns:
        df['yield'] = df['yield'].apply(lambda x: x/100 if x > 20 else x)
        df['yield'] = df['yield'].apply(lambda x: np.nan if x > 100 else x)
        
    # 2. 修正營收成長 (Rev Growth)
    # 同理，若 > 500% 可能是異常，視情況處理，這裡先保留但轉 % 顯示
    
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

# --- 7. 批量掃描 ---
@st.cache_data(ttl=600, show_spinner=False)
def batch_scan_stocks(stock_list, tej_data=None):
    results = []
    
    # 預抓歷史數據算波動率
    try:
        symbols = [s.split(' ')[0] + ('.TW' if not s.endswith('.TW') else '') for s in stock_list]
        hist_data = yf.download(symbols, period="6mo", progress=False)
    except: hist_data = pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_fundamentals, s.split(' ')[0]): s for s in stock_list}
        
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
                history = pd.DataFrame() # 趨勢圖用

                # 波動率 & 趨勢圖數據
                try:
                    s_sym = f"{code}.TW"
                    # 處理 MultiIndex
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        # 嘗試找對應的 Close
                        found = False
                        for col in hist_data.columns:
                            if col[0] == 'Close' and (code in col[1]):
                                closes = hist_data[col].dropna()
                                found = True
                                break
                        if not found: closes = pd.Series()
                    else:
                        closes = hist_data['Close'] if 'Close' in hist_data else pd.Series()

                    if not closes.empty and len(closes) > 10:
                        price = float(closes.iloc[-1]) # 優先用 K 線最新價
                        volatility = closes.pct_change().std() * (252**0.5)
                        history = pd.DataFrame({'Close': closes}) # 存起來畫圖
                except: pass

                # 基本面填入
                if y_data:
                    if pd.isna(price): price = y_data.get('close_price')
                    pe = y_data.get('pe')
                    pb = y_data.get('pb')
                    roe = y_data.get('roe')
                    
                    # 【關鍵修正】單位檢查
                    # Yahoo 有時給 0.03 (3%)，有時給 3.0 (3%)
                    # 我們統一先轉成 % (假設是小數)，後續 sanitize_data 會修正過大的值
                    raw_dy = y_data.get('yield')
                    if raw_dy: dy = raw_dy * 100 
                    
                    raw_rev = y_data.get('rev_growth')
                    if raw_rev: rev_growth = raw_rev * 100
                    
                    peg = y_data.get('peg')
                
                # TEJ 覆蓋
                if tej_data and code in tej_data:
                    t_row = tej_data[code]
                    for k, v in t_row.items():
                        if '法人' in k: chips = float(v) if v != '-' else 0

                # 合成 PEG
                if (pd.isna(peg) or peg == 0) and not pd.isna(pe) and not pd.isna(rev_growth):
                    peg = calculate_synthetic_peg(pe, rev_growth/100)

                # 產業判讀
                industry = 'General'
                if code in ['2330', '2454', '2303', '3034', '3035', '2379', '2382', '3231']: industry = 'Semicon'
                elif code.startswith('28'): industry = 'Finance'
                elif code in ['1101', '1301', '2002', '2603']: industry = 'Cyclical'

                if not pd.isna(price):
                    results.append({
                        '代號': code, '名稱': name, 'close_price': price,
                        'pe': pe, 'pb': pb, 'yield': dy, 'roe': roe,
                        'rev_growth': rev_growth, 'peg': peg, 'chips': chips,
                        'volatility': volatility, 'industry': industry,
                        'history': history
                    })
            except: continue
    
    # 強制建立 DataFrame 避免 KeyError
    df = pd.DataFrame(results)
    required_cols = ['代號', '名稱', 'close_price', 'pe', 'pb', 'yield', 'roe', 'rev_growth', 'peg', 'chips', 'volatility', 'industry', 'history']
    for col in required_cols:
        if col not in df.columns: df[col] = np.nan
        
    return df

# --- 8. 評分與邏輯 ---
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
    
    # ROE 單位處理 (Yahoo 給小數)
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
    
    fill_map = {'pe': 50, 'pb': 5, 'yield': 0, 'rev_growth': 0, 'peg': 5, 'volatility': 0.5, 'roe': 0}
    calc_df = df.fillna(fill_map)

    for idx, row in calc_df.iterrows():
        config = get_sector_config(row.get('industry', 'General'))
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
        buffett_tags.append("🏅" if is_buffett else "")
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

# --- 9. 繪圖與 AI ---
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
        fill='toself', name=title, line_color='#00e676'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                      margin=dict(t=20, b=20, l=20, r=20), height=250)
    return fig

def plot_trend_chart_ui(title, history_df):
    if history_df is None or history_df.empty: return None
    history_df['MA60'] = history_df['Close'].rolling(window=60).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='股價', line=dict(color='#29b6f6')))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['MA60'], name='MA60', line=dict(color='#ffca28', dash='dash')))
    fig.update_layout(title=f"{title} 趨勢", margin=dict(t=30, b=20, l=20, r=20), height=250)
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
重點檢查：
1. **成長爆發力**：營收成長=[REV]%, PEG=[PEG] (若PEG<1.5且成長>20%，請強調為爆發股)。
2. **巴菲特指標**：ROE=[ROE] (是否>15%?), 護城河穩固嗎？
3. **估值風險**：PE=[PE], 殖利率=[YIELD]%。
給出未來6個月操作建議。
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
    use_buffett = st.checkbox("🎩 啟用巴菲特選股邏輯 (價值+護城河)", value=False)
    if use_buffett: st.caption("✅ 已啟用：高 ROE、低波動、低 PEG 標的加分。")

    scan_mode = st.radio("模式選擇", ["🔥 熱門策略", "🏭 產業掃描", "⌨️ 自訂輸入"])
    
    # 預設台股代號字典
    import twstock
    stock_map = {f"{k}{'.TW' if v.market=='上市' else '.TWO'}": f"{k} {v.name}" for k, v in twstock.codes.items() if v.type=='股票'}
    industry_map = {}
    for k, v in twstock.codes.items():
        if v.type=='股票': industry_map.setdefault(v.group, []).append(f"{k}{'.TW' if v.market=='上市' else '.TWO'}")

    target_stocks = []
    if scan_mode == "⌨️ 自訂輸入":
        default = ["2330.TW 台積電", "2317.TW 鴻海", "2454.TW 聯發科", "2881.TW 富邦金"]
        options = sorted(list(stock_map.values()))
        selected = st.multiselect("選擇股票", options, default=[s for s in default if s in options])
        manual = st.text_input("手動輸入代號", "")
        target_stocks = selected
        if manual: target_stocks.append(f"{manual}.TW")
        
    elif scan_mode == "🏭 產業掃描":
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
            raw = batch_scan_stocks(target_stocks, st.session_state['tej_data'])
            # 數據清洗 (Sanitize)
            raw = sanitize_data(raw)
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ 熵值決策選股平台 25.0")
    st.caption("Visual Fix + Data Sanitize + High Contrast UI")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    
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
        st.subheader("🎯 深度戰略分析 (Dashboard)")
        
        for idx, row in final_df.head(10).iterrows():
            with st.container():
                industry_tag = f"<span class='sector-tag'>{row['industry']}</span>"
                buffett_tag = "<span class='buffett-tag'>Buffett Pick</span>" if row['Buffett'] else ""
                
                st.markdown(f"<div class='stock-card'><h3>{row['名稱']} ({row['代號']}) {industry_tag}{buffett_tag}</h3>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 1.5, 1.5])
                
                if idx in df_norm.index:
                    radar_data = get_radar_data(df_norm.loc[idx])
                    with c1:
                        st.plotly_chart(plot_radar_chart_ui(row['名稱'], radar_data), use_container_width=True)
                
                with c2:
                    if 'history' in row and isinstance(row['history'], pd.DataFrame) and not row['history'].empty:
                        st.plotly_chart(plot_trend_chart_ui(row['名稱'], row['history']), use_container_width=True)
                    else:
                        st.warning("無歷史股價數據")

                with c3:
                    st.markdown(f"""
                    #### 關鍵數據
                    - **成長**: 營收成長 {row.get('rev_growth', 'N/A'):.2f}% | PEG {row.get('peg', 'N/A'):.2f}
                    - **價值**: 本益比 {row.get('pe', 'N/A'):.2f} | 殖利率 {row.get('yield', 'N/A'):.2f}%
                    - **風險**: 波動率 {row.get('volatility', 0)*100:.1f}%
                    """)
                    
                    b1, b2 = st.columns(2)
                    if b1.button(f"✨ AI 分析", key=f"ai_{idx}"):
                        p_txt = AI_PROMPT.replace("[STOCK]", row['名稱']).replace("[SECTOR]", str(row['industry'])).replace("[PE]", str(row.get('pe'))).replace("[PEG]", str(row.get('peg'))).replace("[REV]", str(row.get('rev_growth'))).replace("[ROE]", str(row.get('roe')))
                        an = call_ai(p_txt)
                        st.markdown(f"<div class='ai-header'>🤖 AI 觀點</div>{an}", unsafe_allow_html=True)
                    
                    pdf = create_pdf(row.to_dict())
                    b2.download_button("📥 下載報告", pdf, f"{row['代號']}.pdf", key=f"dl_{idx}")
                
                st.markdown("</div>", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請點擊左側「啟動全自動掃描」開始。")
