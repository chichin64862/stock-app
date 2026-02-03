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
from datetime import datetime, timedelta

# --- 1. 專業版介面設定 (FinTech Dark Theme) ---
st.set_page_config(
    page_title="QuantAlpha | 熵值法 x Gemini 戰略分析", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS 強制修正：高對比深色主題 & 能量條樣式 ---
st.markdown("""
<style>
    /* 全局文字顏色修正 */
    body, .stApp, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #e6e6e6 !important;
        font-family: 'Roboto', 'Helvetica Neue', sans-serif;
    }
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* 能量條樣式 */
    .progress-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 2px; }
    .progress-bar-bg { background-color: #30363d; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 10px; }
    .progress-bar-fill { height: 100%; border-radius: 4px; }
    
    /* 因子標籤 */
    .factor-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 5px;
        background-color: #21262d;
        border: 1px solid #30363d;
    }
    
    /* AI Header */
    .ai-header { color: #58a6ff !important; font-weight: bold; font-size: 1.3rem; margin-bottom: 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    
    /* 卡片樣式 */
    .stock-card {
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #30363d; 
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .stock-card:hover {
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 初始化 Session State ---
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'df_norm' not in st.session_state: st.session_state['df_norm'] = None # 用於雷達圖

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

# --- 6. 模型偵測與呼叫 ---
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
    data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    try:
        response = requests.post(url, headers=headers, json=data, proxies=proxies, timeout=60, verify=False)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            try: err_msg = response.json().get('error', {}).get('message', response.text)
            except: err_msg = response.text
            return f"❌ 分析失敗 (Code {response.status_code}): {err_msg}"
    except Exception as e:
        return f"❌ 連線逾時或錯誤: {str(e)}"

# --- 7. 分析提示詞 ---
HEDGE_FUND_PROMPT = """
【角色設定】
你現在是華爾街頂尖的避險基金經理人，專精於「價值投資」與「成長潛力挖掘」。
請針對 **[STOCK]** 進行深度投資分析。

【⚠️ 重要指令】
請務必依據下方提供的 **[最新市場即時數據]** 進行分析。

【最新市場即時數據】
[DATA_CONTEXT]

【分析維度】
1. 訂單能見度 (Revenue Visibility): **重點分析「合約負債」**。若數值很高或有成長，請解讀為未來營收爆發的領先指標。
2. 因子雷達解讀 (Factor Analysis): 根據技術、籌碼、基本面、估值四大面向，指出該股的最強項與最弱項。
3. 綜合決策: 引用最新收盤價，給出「持有」、「買進」或「觀望」建議。
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
        # ... (其他策略可保留，為節省篇幅省略部分重複代碼，邏輯一致) ...
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

# --- 10. 指標與函數 (加入類別標籤) ---
# 這裡我們將指標分類，以便後續繪製雷達圖
indicators_config = {
    'Price vs MA60': {'col': 'priceToMA60', 'direction': '負向', 'name': '季線乖離', 'category': '技術'},
    'Beta': {'col': 'beta', 'direction': '負向', 'name': 'Beta係數', 'category': '技術'},
    'Volume Change': {'col': 'volumeRatio', 'direction': '正向', 'name': '量能比', 'category': '籌碼'},
    'PEG Ratio': {'col': 'pegRatio', 'direction': '負向', 'name': 'PEG', 'category': '估值'},
    'Price To Book': {'col': 'priceToBook', 'direction': '負向', 'name': 'PB比', 'category': '估值'},
    'ROE': {'col': 'returnOnEquity', 'direction': '正向', 'name': 'ROE', 'category': '財報'},
    'Profit Margins': {'col': 'profitMargins', 'direction': '正向', 'name': '淨利率', 'category': '財報'},
}

def fetch_single_stock(ticker):
    try:
        symbol = ticker.split(' ')[0]
        stock = yf.Ticker(symbol)
        info = stock.info 
        
        peg = info.get('pegRatio', None)
        pe = info.get('trailingPE', None)
        growth = info.get('revenueGrowth', 0) 
        if peg is None and pe is not None and growth > 0: peg = pe / (growth * 100)
        elif peg is None: peg = 2.5 
        
        price = info.get('currentPrice', info.get('previousClose', 0))
        ma50 = info.get('fiftyDayAverage', price) 
        bias = (price / ma50) - 1 if ma50 and ma50 > 0 else 0
        beta = info.get('beta', 1.0)
        if beta is None: beta = 1.0
        
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

        return {
            '代號': symbol.replace(".TW", "").replace(".TWO", ""),
            '名稱': info.get('shortName', symbol),
            'close_price': price, 
            'pegRatio': peg, 
            'priceToMA60': bias, 
            'beta': beta,
            'volumeRatio': vol_ratio,
            'priceToBook': info.get('priceToBook', np.nan),
            'returnOnEquity': info.get('returnOnEquity', np.nan), 
            'profitMargins': info.get('profitMargins', np.nan),
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
    if df.empty: return df, None, "No valid data found.", None
    df_norm = df.copy()
    
    # 1. 正規化 (0-1)
    for key, cfg in config.items():
        col = cfg['col']
        mn, mx = df[col].min(), df[col].max()
        denom = mx - mn
        if denom == 0: df_norm[f'{col}_n'] = 0.5
        else:
            if cfg['direction'] == '正向': df_norm[f'{col}_n'] = (df[col] - mn) / denom
            else: df_norm[f'{col}_n'] = (mx - df[col]) / denom
    
    # 2. 熵值權重計算
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
    
    # 3. 計算總分
    df['Score'] = 0
    for key, cfg in config.items():
        df['Score'] += fin_w[key] * df_norm[f'{cfg["col"]}_n'] 
    df['Score'] = (df['Score']*100).round(1)
    
    return df.sort_values('Score', ascending=False), fin_w, None, df_norm

def get_contract_liabilities_safe(symbol_code):
    try:
        if not symbol_code.endswith('.TW') and not symbol_code.endswith('.TWO'): symbol_code += '.TW'
        stock = yf.Ticker(symbol_code)
        bs = stock.balance_sheet
        if bs.empty: return "無財報數據"
        target_keys = ['Contract Liabilities', 'Deferred Revenue', 'Current Contract Liabilities']
        val = None
        for key in target_keys:
            matches = [k for k in bs.index if key in k]
            if matches:
                val = bs.loc[matches[0]].iloc[0]
                break
        if val is not None and not pd.isna(val): return f"{val / 100000000:.2f} 億元"
        else: return "無合約負債數據"
    except: return "讀取失敗"

# --- 輔助函式：繪製雷達圖 ---
def plot_radar_chart(row, df_norm_row, config):
    # 彙整四大面向得分
    categories = {'技術': [], '籌碼': [], '財報': [], '估值': []}
    
    for key, cfg in config.items():
        cat = cfg['category']
        # 取出該指標的正規化得分 (0-1) * 100
        score = df_norm_row[f"{cfg['col']}_n"] * 100
        categories[cat].append(score)
    
    # 計算平均分
    radar_data = {k: np.mean(v) for k, v in categories.items()}
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(radar_data.values()),
        theta=list(radar_data.keys()),
        fill='toself',
        name=row['名稱'],
        line_color='#00e676',
        fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#8b949e'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6e6e6', size=12),
        height=250
    )
    return fig, radar_data

# --- 輔助函式：HTML 能量條 ---
def render_factor_bars(radar_data):
    html = ""
    # 顏色定義
    colors = {'技術': '#29b6f6', '籌碼': '#ab47bc', '財報': '#ffca28', '估值': '#ef5350'}
    
    for cat, score in radar_data.items():
        color = colors.get(cat, '#8b949e')
        # 製作 ■■■■■ 視覺效果
        blocks = int(score / 10) # 10分一格
        visual_bar = "■" * blocks + "░" * (10 - blocks)
        
        html += f"""
        <div style="margin-bottom: 8px;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#e6e6e6;">
                <span><span style="color:{color};">●</span> {cat}</span>
                <span>{score:.0f}%</span>
            </div>
            <div style="font-family: monospace; color:{color}; letter-spacing: 2px;">
                {visual_bar}
            </div>
        </div>
        """
    return html

# --- 11. 儀表板顯示邏輯 ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ QuantAlpha 戰略儀表板 2.0")
    st.caption("Entropy Scoring • Factor Radar • Actionable Timing")
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
    raw = st.session_state['raw_data']
    res, w, err, df_norm = calculate_entropy_score(raw, indicators_config)
    st.session_state['df_norm'] = df_norm # 儲存正規化數據供雷達圖使用
    
    if err: 
        st.error(err)
    else:
        top_n = 10
        top_stocks = res.head(top_n)

        st.markdown("### 🏆 Top 10 潛力標的 (Entropy Ranking)")
        
        # 列表顯示
        st.dataframe(
            top_stocks[['代號', '名稱', 'close_price', 'Score', 'pegRatio', 'priceToMA60', 'beta']],
            column_config={
                "Score": st.column_config.ProgressColumn("Entropy Score", format="%.1f", min_value=0, max_value=100),
                "close_price": st.column_config.NumberColumn("Price", format="%.2f"),
                "pegRatio": st.column_config.NumberColumn("PEG", format="%.2f"),
                "priceToMA60": st.column_config.NumberColumn("MA Bias", format="%.2%"),
            },
            hide_index=True, use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 🎯 深度戰略分析 (Strategic Deep Dive)")
        
        for i, (index, row) in enumerate(top_stocks.iterrows()):
            stock_name = f"{row['代號']} {row['名稱']}"
            is_analyzed = (stock_name in st.session_state['analysis_results'])
            
            # --- 卡片式佈局 ---
            with st.container():
                st.markdown(f"""<div class="stock-card"><h3>{stock_name} <span style="font-size:0.6em;color:#8b949e">NT$ {row['close_price']}</span></h3>""", unsafe_allow_html=True)
                
                # 佈局：左側雷達圖 + 中間因子條 + 右側時機圖
                c1, c2, c3 = st.columns([1.5, 1.2, 2])
                
                # 1. 左側：雷達圖
                with c1:
                    # 抓取該股票的正規化數據
                    norm_row = df_norm.loc[index]
                    fig_radar, radar_data = plot_radar_chart(row, norm_row, indicators_config)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # 2. 中間：因子貢獻度 (文字能量條)
                with c2:
                    st.markdown("**因子貢獻解析**")
                    st.markdown(render_factor_bars(radar_data), unsafe_allow_html=True)
                    
                    # 顯示最強因子
                    best_factor = max(radar_data, key=radar_data.get)
                    st.markdown(f"<div style='margin-top:10px; font-size:0.9rem; color:#00e676;'>🚀 主力優勢: <b>{best_factor}</b></div>", unsafe_allow_html=True)
                
                # 3. 右側：最佳配置時機 (Time Series Trend)
                with c3:
                    st.markdown("**配置時機判定 (Trend vs Value)**")
                    # 模擬繪製股價與均線圖 (這裡需即時抓取歷史數據)
                    try:
                        stock_hist = yf.Ticker(row['代號'].split()[0]).history(period="6mo")
                        if not stock_hist.empty:
                            fig_trend = go.Figure()
                            # 股價線
                            fig_trend.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['Close'], mode='lines', name='Price', line=dict(color='#29b6f6', width=2)))
                            # 標註目前位置
                            last_price = stock_hist['Close'].iloc[-1]
                            fig_trend.add_trace(go.Scatter(x=[stock_hist.index[-1]], y=[last_price], mode='markers', marker=dict(color='#00e676', size=10), name='Current'))
                            
                            # 判斷時機 (簡單邏輯：乖離率負值且分數高 = 最佳買點)
                            timing_msg = "🟢 最佳佈局點 (Value Zone)" if row['priceToMA60'] < 0 else "🟡 持有/觀察 (Momentum)"
                            if row['priceToMA60'] > 0.15: timing_msg = "🔴 留意過熱 (Overheated)"
                            
                            fig_trend.update_layout(
                                title=dict(text=timing_msg, font=dict(size=14, color='#e6e6e6')),
                                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30363d'),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=0,r=0,t=30,b=0), height=250, showlegend=False
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.write("無法取得歷史數據")
                    except:
                        st.write("圖表載入中...")

                # --- AI 分析按鈕 ---
                col_btn, _ = st.columns([1, 4])
                with col_btn:
                     if st.button(f"✨ 生成分析報告", key=f"btn_{i}", use_container_width=True, disabled=is_analyzed):
                         if not is_analyzed:
                            with st.spinner(f"⚡ AI 正在為您撰寫 {stock_name} 的投資備忘錄..."):
                                cl_val = get_contract_liabilities_safe(row['代號'])
                                real_time_data = f"""
                                - 收盤價: {row['close_price']}
                                - 合約負債: {cl_val}
                                - 因子得分: {radar_data} (滿分100)
                                - 季線乖離: {row['priceToMA60']:.2%}
                                """
                                final_prompt = HEDGE_FUND_PROMPT.replace("[STOCK]", stock_name).replace("[DATA_CONTEXT]", real_time_data)
                                result = call_gemini_api(final_prompt)
                                st.session_state['analysis_results'][stock_name] = result
                                st.rerun()
                
                if is_analyzed:
                    st.markdown("<div class='ai-header'>🏛️ Hedge Fund Manager Insight</div>", unsafe_allow_html=True)
                    st.markdown(st.session_state['analysis_results'][stock_name])
                    
                st.markdown("</div>", unsafe_allow_html=True) # End card

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側選擇掃描策略，點擊 **「啟動全自動掃描」** 開始量化分析。")
