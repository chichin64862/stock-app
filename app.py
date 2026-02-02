import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import twstock
import concurrent.futures

# --- 介面設定 ---
st.set_page_config(page_title="熵值法選股 x AI深度分析", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
st.title("🧠 熵值法智能選股 & AI 戰略分析生成器")
st.markdown("### 流程： 1. 量化數據篩選 (Entropy) ➡️ 2. 生成避險基金級提示詞 (Prompt)")

# --- 0. 定義您的超級提示詞模板 ---
HEDGE_FUND_PROMPT = """
【角色設定】
你現在是華爾街頂尖的避險基金經理人，同時具備會計學教授的嚴謹度。請針對 **[STOCK]** 進行深度投資分析。請注意，我不想要模糊的建議，我需要數據支撐的邏輯推演。

【分析維度】
請依序進行以下面向的分析，若需要最新數據請進行聯網搜索：

1. 產業護城河與前景 (Industry & Moat):
綜合最新的大摩、小摩、高盛或台灣本土券商(如富邦、凱基)研究報告，預測該產業未來 6-12 個月的供需狀況。
同業比較 (關鍵)： 比較該公司與同產業競爭對手（列舉 1-2 家）的優劣勢。

2. 籌碼面深度解讀 (Chip Analysis) - 台股極重要:
法人動向： 近期外資與投信（Investment Trust）是連續買超還是賣超？是否有「土洋對作」的情況？
散戶指標： 分析「融資餘額」變化（散戶是否套牢）與「借券賣出餘額」（空軍是否回補）。
大戶持股： 若有數據，請簡述 400 張或 1000 張以上大戶持股比例的趨勢。

3. 技術面狙擊 (Technical Analysis):
趨勢判讀： 結合 K 線型態與均線排列（特別關注季線與月線的乖離率 BIAS）。
關鍵指標：
KD & MACD： 判斷目前是處於背離、鈍化還是黃金/死亡交叉階段？
布林通道： 目前股價位於通道的哪個位置？帶寬 (Bandwidth) 是在壓縮準備變盤，還是已經發散？
成交量結構： 是否出現「價漲量增」的攻擊量，或是「價跌量增」的出貨量？

4. 財務基本面 (Fundamental Deep Dive):
領先指標 - 合約負債： 檢視最新財報「合約負債」或「預收款項」，與 YoY 及 QoQ 相比的變化。
獲利品質： 檢視「營業現金流」是否大於「稅後淨利」？
三率分析： 毛利率、營益率、淨利率的近三季趨勢是向上還是向下？
存貨狀況： 存貨週轉天數是否異常增加？

5. 估值與合理價 (Valuation):
歷史位階： 比較目前的本益比 (PE) 與股價淨值比 (PB) 處於過去 5 年的哪個區間？
PEG 修正： 若為成長股，請評估 PEG (PE / 預估EPS成長率)，PEG < 1 為低估。

【綜合決策與行動指南】
6. 綜合評述 (Executive Summary):
請用一段話總結該股目前的多空力道對比。

7. 實戰操作建議 (Action Plan):
情境 A (空手者)： 若目前想進場，建議的「安全買點」區間在哪裡？(需具體說明回測哪條均線或技術指標位置)
情境 B (持股者)： 建議的「停利點」或「停損點」應設在哪個技術關卡？
風險提示： 未來 3 個月最大的下檔風險是什麼？
"""

# --- 1. 自動建立台股清單 ---
@st.cache_data
def get_tw_stock_list():
    codes = twstock.codes
    stock_list = []
    for code, info in codes.items():
        if info.type == '股票':
            if info.market == '上市': suffix = '.TW'
            elif info.market == '上櫃': suffix = '.TWO'
            else: continue
            stock_list.append(f"{code}{suffix} {info.name}")
    return sorted(stock_list)

all_stocks = get_tw_stock_list()

# --- 2. 設定區 ---
with st.expander("🔍 步驟一：建立股票池 (可搜尋)", expanded=True):
    default_selection = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海", "2603.TW 長榮", "3034.TW 聯詠", "2382.TW 廣達", "3231.TW 緯創"]
    selected_items = st.multiselect(
        "選擇要分析的股票 (建議選 10-20 檔進行排名):",
        options=all_stocks,
        default=[s for s in default_selection if s in all_stocks]
    )
    run_btn = st.button("🚀 開始熵值運算", type="primary", use_container_width=True)

# --- 3. 指標設定 ---
indicators_config = {
    'Trailing PE': {'col': 'trailingPE', 'direction': '負向', 'name': '本益比 (PE)'},
    'Price To Book': {'col': 'priceToBook', 'direction': '負向', 'name': '股價淨值比 (PB)'},
    'ROE': {'col': 'returnOnEquity', 'direction': '正向', 'name': 'ROE'},
    'Profit Margins': {'col': 'profitMargins', 'direction': '正向', 'name': '淨利率'},
    'Revenue Growth': {'col': 'revenueGrowth', 'direction': '正向', 'name': '營收成長'},
    'Dividend Yield': {'col': 'dividendRate', 'direction': '正向', 'name': '殖利率'},
    'Debt to Equity': {'col': 'debtToEquity', 'direction': '負向', 'name': '負債比'}
}

# --- 核心函數：抓取單一股票 ---
def fetch_single_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info 
        
        div = info.get('dividendYield', 0)
        if div is None: div = 0
        
        return {
            '代號': ticker.replace(".TW", "").replace(".TWO", ""),
            '名稱': info.get('shortName', ticker),
            'trailingPE': info.get('trailingPE', np.nan),
            'priceToBook': info.get('priceToBook', np.nan),
            'returnOnEquity': info.get('returnOnEquity', np.nan),
            'profitMargins': info.get('profitMargins', np.nan),
            'revenueGrowth': info.get('revenueGrowth', np.nan),
            'dividendRate': div,
            'debtToEquity': info.get('debtToEquity', np.nan)
        }
    except:
        return None

# --- 多工抓取 ---
def get_stock_data_concurrent(selected_list):
    tickers = [item.split(' ')[0] for item in selected_list]
    data = []
    progress_bar = st.progress(0, text="啟動多工引擎，下載數據中...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_stock, t): t for t in tickers}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result: data.append(result)
            completed += 1
            progress_bar.progress(completed / len(tickers), text=f"已下載 {completed}/{len(tickers)}")
            
    return pd.DataFrame(data)

# --- 熵值計算 (已修正 KeyError) ---
def calculate_entropy_score(df, config):
    df = df.dropna().copy()
    if df.empty: return df, None, "有效數據不足 (可能有缺漏值)"
    df_norm = df.copy()
    
    # 1. 標準化
    for key, cfg in config.items():
        col = cfg['col']
        if col == 'trailingPE': df[col] = df[col].apply(lambda x: x if x > 0 else df[col].max())
        
        mn, mx = df[col].min(), df[col].max()
        denom = mx - mn
        if denom == 0: df_norm[f'{col}_n'] = 0.5
        else:
            if cfg['direction'] == '正向': df_norm[f'{col}_n'] = (df[col] - mn) / denom
            else: df_norm[f'{col}_n'] = (mx - df[col]) / denom
            
    m = len(df)
    k = 1 / np.log(m) if m > 1 else 0
    weights = {}
    
    # 2. 計算權重 (Key 統一使用 config key，例如 'Trailing PE')
    for key, cfg in config.items():
        col = cfg['col']
        p = df_norm[f'{col}_n'] / df_norm[f'{col}_n'].sum() if df_norm[f'{col}_n'].sum() != 0 else 0
        e = -k * np.sum(p * np.log(p + 1e-9))
        weights[key] = 1 - e  # <--- 修正點：使用 key 而不是 col
        
    tot = sum(weights.values())
    fin_w = {k: v/tot for k, v in weights.items()}
    
    # 3. 計算總分
    df['Score'] = 0
    for key, cfg in config.items():
        df['Score'] += fin_w[key] * df_norm[f'{cfg["col"]}_n'] # <--- 修正點：使用 fin_w[key]
    
    df['Score'] = (df['Score']*100).round(1)
    return df.sort_values('Score', ascending=False), fin_w, None

# --- 主執行區 ---
if run_btn:
    if not selected_items:
        st.warning("⚠️ 請先選擇股票！")
    else:
        # 1. 計算排名
        raw = get_stock_data_concurrent(selected_items)
        if not raw.empty:
            res, w, err = calculate_entropy_score(raw, indicators_config)
            if err: 
                st.error(err)
            else:
                # 2. 顯示排名表
                st.markdown("---")
                col_res, col_chart = st.columns([2, 1])
                
                with col_res:
                    st.subheader("📊 熵值法綜合排名 (前 5 名)")
                    top_5 = res.head(5)
                    st.dataframe(
                        top_5[['名稱', '代號', 'Score', 'trailingPE', 'priceToBook', 'returnOnEquity', 'profitMargins']]
                        .style.background_gradient(subset=['Score'], cmap='Greens')
                        .format({'returnOnEquity': '{:.1%}', 'profitMargins': '{:.1%}', 'priceToBook': '{:.2f}'}),
                        use_container_width=True
                    )
                
                with col_chart:
                    st.subheader("⚖️ AI 權重計算結果")
                    # 這裡 w[k] 現在可以正確運作了，因為 w 的 key 已經修正為 config key
                    w_df = pd.DataFrame([{'指標':v['name'], '權重':w[k]} for k,v in indicators_config.items()])
                    st.plotly_chart(px.pie(w_df, values='權重', names='指標'), use_container_width=True)

                # 3. 生成深度分析提示詞
                st.markdown("---")
                st.header("🤖 步驟二：AI 深度分析指令 (Top 5)")
                st.info("👇 點擊下方的「複製按鈕」，直接貼給 ChatGPT / Gemini / Claude 進行分析！")

                for index, row in top_5.iterrows():
                    stock_name = f"{row['代號']} {row['名稱']}"
                    final_prompt = HEDGE_FUND_PROMPT.replace("[STOCK]", stock_name)
                    
                    with st.expander(f"🏆 第 {index+1} 名：{stock_name} (點擊展開複製)", expanded=(index==0)):
                        st.text_area(f"給 AI 的指令 ({stock_name})", value=final_prompt, height=200, key=f"p_{index}")
                        st.markdown(f"**建議指令：** 複製上方內容，發送給 AI 即可獲得避險基金級報告。")

        else:
            st.error("無法獲取數據，請稍後再試。")
