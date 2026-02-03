import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import twstock
import concurrent.futures
import requests
import json
import time
import os

# --- 介面設定 ---
st.set_page_config(page_title="熵值法 x Gemini 全自動分析", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 熵值法選股 & Gemini 全自動戰略分析")
st.markdown("### 流程： 1. 自動掃描選股 ➡️ 2. Gemini API 即時撰寫報告")

# --- 0. 初始化 Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {}
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state:
    st.session_state['scan_finished'] = False

# --- 1. 安全讀取 API Key (從 Secrets) ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ 系統偵測不到 API Key！請確認您已在 Streamlit Cloud > Settings > Secrets 中設定 `GEMINI_API_KEY`。")
    st.stop()

# --- 2. 環境設定 (Proxy 與 SSL) ---
proxies = {}
if os.getenv("HTTP_PROXY"): proxies["http"] = os.getenv("HTTP_PROXY")
if os.getenv("HTTPS_PROXY"): proxies["https"] = os.getenv("HTTPS_PROXY")

# --- 3. 核心功能：自動偵測可用模型 ---
def get_available_model(key):
    # 預設使用 flash，因為它最快且最不容易卡住
    default_model = "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, proxies=proxies, timeout=5, verify=False)
        if response.status_code == 200:
            data = response.json()
            # 優先找 Flash
            for m in data.get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'flash' in m['name']:
                    return m['name'].replace('models/', '')
            # 次要找 Pro
            for m in data.get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []) and 'pro' in m['name']:
                    return m['name'].replace('models/', '')
    except:
        pass
    return default_model

# --- 4. 呼叫 Gemini API (REST 方式) ---
def call_gemini_api(prompt):
    target_model = get_available_model(api_key)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    
    try:
        # 設定 60 秒超時，避免無限轉圈
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

# --- 定義分析提示詞 ---
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

# --- 5. 數據與清單處理 ---
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

# --- 6. 側邊欄：掃描模式選擇 ---
with st.sidebar:
    st.header("🎛️ 掃描控制台")
    scan_mode = st.radio("選股模式：", ["自行輸入/多選", "🔥 熱門策略掃描", "🏭 產業類股掃描"])
    target_stocks = []
    
    if scan_mode == "自行輸入/多選":
        default_selection = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海"]
        selected = st.multiselect("選擇股票:", options=sorted(list(stock_map.values())), default=[s for s in default_selection if s in stock_map.values()])
        target_stocks = selected
        st.caption(f"已選擇 {len(target_stocks)} 檔股票")
        
    elif scan_mode == "🔥 熱門策略掃描":
        strategy = st.selectbox("選擇策略:", ["台灣50成份股 (大型權值)", "中型100成份股 (成長潛力)", "高股息熱門股 (存股族)", "AI 供應鏈概念", "貨櫃航運三雄"])
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
        
        st.info(f"已載入【{strategy}】清單，共 {len(target_stocks)} 檔。請點擊下方按鈕開始分析。")

    elif scan_mode == "🏭 產業類股掃描":
        all_industries = sorted(list(industry_map.keys()))
        selected_industry = st.selectbox("選擇產業:", all_industries)
        if selected_industry:
            codes = industry_map[selected_industry]
            target_stocks = [stock_map[c] for c in codes if c in stock_map]
            st.info(f"已鎖定【{selected_industry}】，共 {len(target_stocks)} 檔。請點擊下方按鈕開始分析。")
            if len(target_stocks) > 60: st.warning("⚠️ 數量較多，掃描時間可能較長。")
    
    run_btn = st.button("🚀 啟動全自動掃描", type="primary", use_container_width=True)

# --- 7. 指標與函數 ---
indicators_config = {
    'PEG Ratio': {'col': 'pegRatio', 'direction': '負向', 'name': 'PEG (估值成長比)'},
    'ROE': {'col': 'returnOnEquity', 'direction': '正向', 'name': 'ROE'},
    'Profit Margins': {'col': 'profitMargins', 'direction': '正向', 'name': '淨利率'},
    'Price vs MA60': {'col': 'priceToMA60', 'direction': '正向', 'name': '季線乖離率'},
    'Price To Book': {'col': 'priceToBook', 'direction': '負向', 'name': 'PB'},
    'Dividend Yield': {'col': 'dividendRate', 'direction': '正向', 'name': '殖利率'}
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
    progress_bar = st.progress(0, text="正在喚醒 AI 掃描引擎...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(fetch_single_stock, t): t for t in selected_list}
        completed = 0
        total = len(selected_list)
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result: data.append(result)
            completed += 1
            progress_bar.progress(completed / total, text=f"已掃描 {completed}/{total} 檔...")
    return pd.DataFrame(data)

def calculate_entropy_score(df, config):
    df = df.dropna().copy()
    if df.empty: return df, None, "有效數據不足"
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

# --- 主執行區 ---
if run_btn:
    if not target_stocks:
        st.warning("⚠️ 請先選擇至少一檔股票或一個策略！")
    else:
        st.session_state['analysis_results'] = {}
        st.session_state['raw_data'] = None
        raw = get_stock_data_concurrent(target_stocks)
        if not raw.empty:
            st.session_state['raw_data'] = raw
            st.session_state['scan_finished'] = True
            st.rerun()

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    raw = st.session_state['raw_data']
    st.markdown("---")
    
    res, w, err = calculate_entropy_score(raw, indicators_config)
    
    if err: 
        st.error(err)
    else:
        top_n = 10
        st.subheader(f"🏆 掃描結果：前 {top_n} 強潛力股")
        top_stocks = res.head(top_n)
        st.dataframe(
            top_stocks[['名稱', '代號', 'Score', 'pegRatio', 'priceToMA60', 'returnOnEquity', 'profitMargins']]
            .style.background_gradient(subset=['Score'], cmap='Greens')
            .format({'returnOnEquity': '{:.1%}', 'profitMargins': '{:.1%}', 'pegRatio': '{:.2f}', 'priceToMA60': '{:.2%}'}),
            use_container_width=True
        )
        
        st.markdown("---")
        st.header(f"🤖 Gemini AI 深度分析 (點擊按鈕即時生成)")
        
        for i, (index, row) in enumerate(top_stocks.iterrows()):
            stock_name = f"{row['代號']} {row['名稱']}"
            final_prompt = HEDGE_FUND_PROMPT.replace("[STOCK]", stock_name)
            
            is_analyzed = (stock_name in st.session_state['analysis_results'])
            
            with st.expander(f"🏆 第 {i+1} 名：{stock_name} (分數: {row['Score']})", expanded=(i==0 or is_analyzed)):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if is_analyzed:
                        st.success("✅ 分析報告已生成")
                    else:
                        st.caption("AI 分析核心指令已準備就緒...")
                        
                with col2:
                    # 【核心修改】改用直接的 if 判斷，避免 callback 卡住
                    if st.button(f"✨ AI 分析", key=f"btn_{i}", use_container_width=True):
                        if not is_analyzed:
                            with st.spinner(f"🤖 AI 正在深入分析 {stock_name} 的財報與籌碼，請稍候約 15 秒..."):
                                result = call_gemini_api(final_prompt)
                                st.session_state['analysis_results'][stock_name] = result
                                st.toast(f"✅ {stock_name} 分析完成！")
                                time.sleep(0.5)
                                st.rerun() # 強制刷新畫面，顯示結果

                if is_analyzed:
                    st.markdown("### 📝 AI 分析報告")
                    st.markdown(st.session_state['analysis_results'][stock_name])

elif not st.session_state['scan_finished']:
    st.info("👈 請在左側選擇選股模式與範圍，確認無誤後點擊「啟動全自動掃描」按鈕。")
