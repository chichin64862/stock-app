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

# --- 1. 專業版介面設定 (FinTech Dark Theme) ---
st.set_page_config(
    page_title="QuantAlpha | 熵值法 x Gemini 戰略分析", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 強制修正：高對比深色主題 ---
st.markdown("""
<style>
    /* 全局文字顏色修正 */
    body, .stApp, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #e6e6e6 !important;
        font-family: 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    /* 背景色設定 */
    .stApp {
        background-color: #0e1117;
    }
    
    /* 側邊欄優化 */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #58a6ff !important;
    }
    
    /* 指標卡片 (Metric) */
    div[data-testid="stMetric"] {
        background-color: #21262d;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    div[data-testid="stMetricValue"] {
        color: #2ea043 !important; /* 漲跌色 */
    }
    
    /* 表格 (DataFrame) 文字修正 */
    div[data-testid="stDataFrame"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 5px;
    }
    
    /* 按鈕樣式 */
    div.stButton > button {
        background-color: #238636;
        color: white !important;
        border: 1px solid #rgba(255,255,255,0.1);
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
        border-color: #f0f6fc;
    }
    
    /* Expander (摺疊區塊) */
    .streamlit-expanderHeader {
        background-color: #21262d;
        color: #e6e6e6 !important;
        border-radius: 5px;
    }
    
    /* AI 分析標題 */
    .ai-header {
        color: #58a6ff !important;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 12px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }
    
    /* 分數解釋區塊 */
    .score-legend {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.9rem;
        border-left: 4px solid #a371f7;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 初始化 Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state:
    st.session_state['scan_finished'] = False

# --- 4. 安全讀取 API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！請確認您已在 Streamlit Cloud > Settings > Secrets 中設定 `GEMINI_API_KEY`。")
    st.stop()

# --- 5. 環境與連線設定 ---
proxies = {}
if os.getenv("HTTP_PROXY"): proxies["http"] = os.getenv("HTTP_PROXY")
if os.getenv("HTTPS_PROXY"): proxies["https"] = os.getenv("HTTPS_PROXY")

# --- 6. 模型偵測與呼叫 (邏輯不變) ---
def get_available_model(key):
    default_model = "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, proxies=proxies, timeout=5, verify=False)
        if response.status_code == 200:
            data = response.json()
            for m in data.get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'flash' in m['name']:
                    return m['name'].replace('models/', '')
            for m in data.get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'pro' in m['name']:
                    return m['name'].replace('models/', '')
    except:
        pass
    return default_model

def call_gemini_api(prompt):
    target_model = get_available_model(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    try:
        response = requests.post(url, headers=headers, json=data, proxies=proxies, timeout=60, verify=False)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            try:
                err_msg = response.json().get('error', {}).get('message', response.text)
            except:
                err_msg = response.text
            return f"❌ 分析失敗 (Code {response.status_code}): {err_msg}"
    except Exception as e:
        return f"❌ 連線逾時或錯誤: {str(e)}"

# --- 7. 分析提示詞 (不變) ---
HEDGE_FUND_PROMPT = """
【角色設定】
你現在是華爾街頂尖的避險基金經理人，同時具備會計學教授的嚴謹度。請針對 **[STOCK]** 進行深度投資分析。

【分析維度】
1. 產業護城河與前景 (Industry & Moat): 預測未來 6-12 個月供需。比較同業優劣。
2. 籌碼面深度解讀 (Chip Analysis): 外資投信動向、融資融券變化(若無具體數據請根據股價型態推論)。
3. 技術面狙擊 (Technical Analysis): 季線乖離率(MA60)、KD/MACD 背離、成交量結構。
4. 財務基本面 (Fundamental): 合約負債變化、營運現金流vs淨利、三率趨勢、存貨週轉。
5. 估值 (Valuation): 本益比/股價淨值比歷史區間、PEG 評估。

【綜合決策】
6. 總結與實戰建議: 給出空手者「安全買點」與持股者「停利停損點」。風險提示。
"""

# --- 8. 數據與清單處理 ---
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
            if industry not in industry_dict:
                industry_dict[industry] = []
            industry_dict[industry].append(full_code)
    return stock_dict, industry_dict

stock_map, industry_map = get_tw_stock_info()

# --- 9. 側邊欄設定 ---
with st.sidebar:
    st.title("🎛️ QuantAlpha 控制台")
    st.markdown("---")
    
    st.subheader("1️⃣ 篩選範圍 (Universe)")
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
            target_stocks = []
            for c in codes:
                if f"{c}.TW" in stock_map: target_stocks.append(stock_map[f"{c}.TW"])
                elif f"{c}.TWO" in stock_map: target_stocks.append(stock_map[f"{c}.TWO"])
        elif strategy == "貨櫃航運三雄":
            target_stocks = ["2603.TW 長榮", "2609.TW 陽明", "2615.TW 萬海"]
        st.info(f"已載入 {len(target_stocks)} 檔標的")

    elif scan_mode == "🏭 產業類股掃描":
        all_industries = sorted(list(industry_map.keys()))
        selected_industry = st.selectbox("選擇產業:", all_industries)
        if selected_industry:
            codes = industry_map[selected_industry]
            target_stocks = [stock_map[c] for c in codes if c in stock_map]
            st.info(f"鎖定 {len(target_stocks)} 檔標的")

    st.markdown("---")
    st.subheader("2️⃣ 執行掃描")
    run_btn = st.button("🚀 啟動全自動掃描", type="primary", use_container_width=True)

# --- 10. 指標與函數 ---
indicators_config = {
    'PEG Ratio': {'col': 'pegRatio', 'direction': '負向', 'name': 'PEG (成長估值)'},
    'ROE': {'col': 'returnOnEquity', 'direction': '正向', 'name': 'ROE (權益報酬)'},
    'Profit Margins': {'col': 'profitMargins', 'direction': '正向', 'name': '淨利率 (獲利力)'},
    'Price vs MA60': {'col': 'priceToMA60', 'direction': '正向', 'name': '季線乖離 (趨勢)'},
    'Price To Book': {'col': 'priceToBook', 'direction': '負向', 'name': 'PB (股價淨值)'},
    'Dividend Yield': {'col': 'dividendRate', 'direction': '正向', 'name': '殖利率 (防禦)'}
}

def fetch_single_stock(ticker):
    try:
        symbol = ticker.split(' ')[0]
        stock = yf.Ticker(symbol)
        info = stock.info 
        peg = info.get('pegRatio', None)
        pe = info.get('trailingPE', None)
        growth = info.get('revenueGrowth', 0) 
        if peg is None and pe is not None and growth > 0:
            peg = pe / (growth * 100)
        elif peg is None: peg = 2.5 
        price = info.get('currentPrice', info.get('previousClose', 0))
        ma50 = info.get('fiftyDayAverage', price) 
        bias = (price / ma50) - 1 if ma50 and ma50 > 0 else 0
        div = info.get('dividendYield', 0)
        return {
            '代號': symbol.replace(".TW", "").replace(".TWO", ""),
            '名稱': info.get('shortName', symbol),
            'pegRatio': peg, 'priceToMA60': bias, 'priceToBook': info.get('priceToBook', np.nan),
            'returnOnEquity': info.get('returnOnEquity', np.nan), 'profitMargins': info.get('profitMargins', np.nan),
            'dividendRate': div if div else 0
        }
    except: return None

def get_stock_data_concurrent(selected_list):
    data = []
    progress_bar = st.progress(0, text="Initializing Quantitative Scanner...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(fetch_single_stock, t): t for t in selected_list}
        completed = 0
        total = len(selected_list)
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result: data.append(result)
            completed += 1
            progress_bar.progress(completed / total, text=f"Scanning Market Data: {completed}/{total}...")
    progress_bar.empty()
    return pd.DataFrame(data)

def calculate_entropy_score(df, config):
    df = df.dropna().copy()
    if df.empty: return df, None, "No valid data found."
    df_norm = df.copy()
    for key, cfg in config.items():
        col = cfg['col']
        mn, mx = df[col].min(), df[col].max()
        denom = mx - mn
        if denom == 0: df_norm[f'{col}_n'] = 0.5
        else:
            if cfg['direction'] == '正向': df_norm[f'{col}_n'] = (df[col] - mn) / denom
            else: df_norm[f'{col}_n'] = (mx - df[col]) / denom
    m = len(df)
    k = 1 / np.log(m) if m > 1 else 0
    weights = {}
    for key, cfg in config.items():
        col = cfg['col']
        p = df_norm[f'{col}_n'] / df_norm[f'{col}_n'].sum() if df_norm[f'{col}_n'].sum() != 0 else 0
        e = -k * np.sum(p * np.log(p + 1e-9))
        weights[key] = 1 - e 
    tot = sum(weights.values())
    fin_w = {k: v/tot for k, v in weights.items()}
    df['Score'] = 0
    for key, cfg in config.items():
        df['Score'] += fin_w[key] * df_norm[f'{cfg["col"]}_n'] 
    df['Score'] = (df['Score']*100).round(1)
    return df.sort_values('Score', ascending=False), fin_w, None

# --- 11. 儀表板顯示邏輯 ---

# 頂部標題與解釋
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📈 QuantAlpha 智慧選股終端")
    st.caption("Entropy Method Selection • Gemini AI Insights • Real-time Data")
with col2:
    if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
         st.metric("Total Scanned", f"{len(st.session_state['raw_data'])} Stocks", delta="Live Update")

# 執行掃描邏輯
if run_btn:
    if not target_stocks:
        st.warning("⚠️ Please select at least one stock or strategy from the sidebar.")
    else:
        st.session_state['analysis_results'] = {}
        st.session_state['raw_data'] = None
        raw = get_stock_data_concurrent(target_stocks)
        if not raw.empty:
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

# 結果顯示區
if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    raw = st.session_state['raw_data']
    res, w, err = calculate_entropy_score(raw, indicators_config)
    
    if err: 
        st.error(err)
    else:
        top_n = 10
        top_stocks = res.head(top_n)

        # --- 新增：模型原理與分數解釋 ---
        with st.expander("ℹ️ 關於熵值模型分數 (Entropy Score) 的定義", expanded=True):
            st.markdown("""
            <div class='score-legend'>
                <h4>🧮 什麼是熵值評分 (Entropy Score)?</h4>
                <p>這不是主觀評分，而是透過<b>「資訊熵 (Information Entropy)」</b>演算法計算出來的客觀權重。</p>
                <ul>
                    <li><b>原理</b>：演算法會自動判斷哪些指標（如 ROE、PEG）在目前這群股票中差異最大。差異越大的指標，代表提供的資訊量越多，獲得的權重就越高。</li>
                    <li><b>50 分代表什麼？</b>：50 分代表該股票在所有指標的綜合表現上，處於<b>「平均水準」</b>。</li>
                </ul>
                <hr style='border-color: #30363d;'>
                <b>📊 分數解讀指南：</b>
                <ul>
                    <li><span style='color:#2ea043; font-weight:bold;'>🟢 > 60 分 (優異)</span>：該股票在關鍵指標上顯著優於同業，屬於強勢潛力股。</li>
                    <li><span style='color:#e3b341; font-weight:bold;'>🟡 40 - 60 分 (中性)</span>：各項數據表現平均，無明顯短版但亮點不足。</li>
                    <li><span style='color:#f85149; font-weight:bold;'>🔴 < 40 分 (落後)</span>：財務或技術指標低於群體平均，需留意風險。</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- Section 1: Market Overview & Entropy Analysis ---
        st.markdown("### 📊 市場熵值模型分析 (Entropy Market Model)")
        
        c1, c2 = st.columns([1.8, 1.2])
        
        with c1:
            st.markdown("**Top Ranked Assets (Entropy Score)**")
            st.dataframe(
                top_stocks[['代號', '名稱', 'Score', 'pegRatio', 'returnOnEquity', 'profitMargins']],
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Entropy Score",
                        help="綜合評分 (0-100)，越高越好",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    "pegRatio": st.column_config.NumberColumn("PEG", format="%.2f"),
                    "returnOnEquity": st.column_config.NumberColumn("ROE", format="%.1f%%"),
                    "profitMargins": st.column_config.NumberColumn("Net Margin", format="%.1f%%"),
                },
                hide_index=True,
                use_container_width=True,
                height=350
            )

        with c2:
            st.markdown("**Factor Weight Distribution (演算法自動賦權)**")
            w_df = pd.DataFrame(list(w.items()), columns=['Factor', 'Weight'])
            # 使用 plotly_dark 主題解決黑底看不見圖的問題
            fig = px.bar(w_df, x='Weight', y='Factor', orientation='h', 
                         title="Entropy Calculated Weights",
                         text_auto='.1%', color='Weight', template='plotly_dark')
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e6e6e6',
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=True, gridcolor='#30363d'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Section 2: AI Strategic Analysis ---
        st.markdown("---")
        st.markdown("### 🤖 Gemini AI 戰略分析中心 (Strategic Intelligence)")
        
        for i, (index, row) in enumerate(top_stocks.iterrows()):
            stock_name = f"{row['代號']} {row['名稱']}"
            final_prompt = HEDGE_FUND_PROMPT.replace("[STOCK]", stock_name)
            is_analyzed = (stock_name in st.session_state['analysis_results'])
            
            # 使用 container 模擬卡片效果
            with st.container():
                st.markdown(f"""
                <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: #58a6ff;">{stock_name}</h3>
                            <span style="color: #8b949e; font-size: 0.9em;">Entropy Score: <b>{row['Score']}</b> | ROE: <b>{row['returnOnEquity']:.1%}</b></span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_btn, col_status = st.columns([1, 4])
                with col_btn:
                     if st.button(f"✨ 生成分析報告", key=f"btn_{i}", use_container_width=True, disabled=is_analyzed):
                         if not is_analyzed:
                            with st.spinner(f"⚡ 正在連線 Gemini AI 深度分析 {stock_name}..."):
                                result = call_gemini_api(final_prompt)
                                st.session_state['analysis_results'][stock_name] = result
                                st.rerun()
                
                if is_analyzed:
                    st.markdown("<div class='ai-header'>🏛️ Hedge Fund Manager Insight</div>", unsafe_allow_html=True)
                    st.markdown(st.session_state['analysis_results'][stock_name])
                    st.markdown("---")
                    st.caption(f"Generated by Google Gemini 1.5 • Confidence Level: High • Data as of {time.strftime('%Y-%m-%d')}")
                    st.markdown("<br>", unsafe_allow_html=True)

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側選擇掃描策略，點擊 **「啟動全自動掃描」** 開始量化分析。")
