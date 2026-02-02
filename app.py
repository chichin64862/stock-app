%%writefile app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import twstock
import concurrent.futures # 引入多工處理模組

# --- 介面設定 ---
st.set_page_config(page_title="台股極速選股", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.title("⚡ 熵值法智能選股 (極速版)")

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
with st.expander("🔍 建立您的股票池", expanded=True):
    default_selection = ["2330.TW 台積電", "2454.TW 聯發科", "2317.TW 鴻海", "2603.TW 長榮", "2609.TW 陽明"]
    selected_items = st.multiselect(
        "選擇股票 (建議可選 10-20 檔測試速度):",
        options=all_stocks,
        default=[s for s in default_selection if s in all_stocks]
    )
    run_btn = st.button("🚀 極速分析", type="primary", use_container_width=True)

# 指標設定
indicators_config = {
    'Trailing PE': {'col': 'trailingPE', 'direction': '負向', 'name': '本益比'},
    'ROE': {'col': 'returnOnEquity', 'direction': '正向', 'name': 'ROE'},
    'Revenue Growth': {'col': 'revenueGrowth', 'direction': '正向', 'name': '營收成長'},
    'Dividend Yield': {'col': 'dividendRate', 'direction': '正向', 'name': '殖利率'},
    'Debt to Equity': {'col': 'debtToEquity', 'direction': '負向', 'name': '負債比'}
}

# --- 核心優化：單一股票抓取函數 ---
def fetch_single_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        # fast_info 通常比 .info 快，但資料較少，這裡還是用 .info 確保資料完整，但透過多工加速
        info = stock.info 
        
        div = info.get('dividendYield', 0)
        if div is None: div = 0
        
        return {
            '代號': ticker.replace(".TW", "").replace(".TWO", ""),
            '名稱': info.get('shortName', ticker),
            'trailingPE': info.get('trailingPE', np.nan),
            'returnOnEquity': info.get('returnOnEquity', np.nan),
            'revenueGrowth': info.get('revenueGrowth', np.nan),
            'dividendRate': div,
            'debtToEquity': info.get('debtToEquity', np.nan)
        }
    except:
        return None

# --- 核心優化：多工抓取主函數 ---
def get_stock_data_concurrent(selected_list):
    tickers = [item.split(' ')[0] for item in selected_list]
    data = []
    
    # 顯示進度條
    progress_bar = st.progress(0, text="啟動多工引擎...")
    
    # 使用 ThreadPoolExecutor 同時抓取 (預設開 10 個執行緒)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 送出所有任務
        future_to_ticker = {executor.submit(fetch_single_stock, t): t for t in tickers}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result:
                data.append(result)
            
            completed_count += 1
            progress_bar.progress(completed_count / len(tickers), text=f"已完成 {completed_count}/{len(tickers)}")
            
    return pd.DataFrame(data)

# 熵值計算 (維持不變)
def calculate_entropy_score(df, config):
    df = df.dropna().copy()
    if df.empty: return df, None, "數據不足或有缺失值"
    df_norm = df.copy()
    
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
    for key, cfg in config.items():
        col = cfg['col']
        p = df_norm[f'{col}_n'] / df_norm[f'{col}_n'].sum() if df_norm[f'{col}_n'].sum() != 0 else 0
        e = -k * np.sum(p * np.log(p + 1e-9))
        weights[col] = 1 - e
        
    tot = sum(weights.values())
    if tot == 0: return df, None, "無法計算權重"
    fin_w = {k: v/tot for k, v in weights.items()}
    
    df['Score'] = 0
    for key, cfg in config.items():
        df['Score'] += fin_w[cfg['col']] * df_norm[f'{cfg["col"]}_n']
    df['Score'] = (df['Score']*100).round(1)
    return df.sort_values('Score', ascending=False), fin_w, None

# --- 主執行區 ---
if run_btn:
    if not selected_items:
        st.warning("⚠️ 請選擇股票！")
    else:
        with st.spinner('⚡ 極速運算中...'):
            raw = get_stock_data_concurrent(selected_items)
            if not raw.empty:
                res, w, err = calculate_entropy_score(raw, indicators_config)
                if err: st.error(err)
                else:
                    top = res.iloc[0]
                    st.balloons()
                    st.success(f"🏆 冠軍：**{top['名稱']}** | 分數：{top['Score']}")
                    
                    st.dataframe(
                        res[['名稱', '代號', 'Score', 'trailingPE', 'returnOnEquity', 'dividendRate']]
                        .style.background_gradient(subset=['Score'], cmap='Greens')
                        .format({'dividendRate': '{:.2%}', 'returnOnEquity': '{:.2%}'}),
                        use_container_width=True
                    )
                    
                    with st.expander("📊 權重分析"):
                        w_df = pd.DataFrame([{'指標':v['name'], '權重':w[k]} for k,v in indicators_config.items()])
                        st.plotly_chart(px.pie(w_df, values='權重', names='指標'), use_container_width=True)
            else:
                st.error("無法獲取數據")