import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
import os
from datetime import datetime
import urllib.request
import html
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- 1. 介面設定與 CSS ---
st.set_page_config(page_title="台股量化與價值分析終端", page_icon="Terminal", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    :root { color-scheme: dark !important; }
    .stApp { background-color: #06090F !important; }
    [data-testid="stSidebar"] { background-color: #0D131F !important; border-right: 1px solid #1E293B !important; }
    h1, h2, h3, p, span, div, label { font-family: 'Segoe UI', Tahoma, sans-serif; color: #E2E8F0 !important; }
    .stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] > div, input { background-color: #0D131F !important; color: white !important; border: 1px solid #1E293B !important; }
    .stMultiSelect span[data-baseweb="tag"] { background-color: #1E293B !important; color: #F8FAFC !important; }
    .stock-card { background-color: #0D131F !important; padding: 16px 24px; border-radius: 4px; border: 1px solid #1E293B !important; border-left: 4px solid #3B82F6 !important; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .card-header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid #1E293B !important; padding-bottom: 12px; margin-bottom: 16px; }
    .header-title { font-size: 1.4rem; font-weight: 700; color: #FFFFFF !important; }
    .header-price { font-size: 1.4rem; font-weight: 700; color: #10B981 !important; margin-left: 16px; font-family: 'Consolas', monospace; }
    .quote-board { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background-color: #1E293B !important; border-radius: 4px; }
    .quote-item { background-color: #06090F !important; padding: 12px 16px; display: flex; flex-direction: column; justify-content: center; }
    .q-label { color: #64748B !important; font-size: 0.75rem; font-weight: 600; margin-bottom: 4px; }
    .q-val { color: #F8FAFC !important; font-weight: 700; font-size: 1.15rem; font-family: 'Consolas', monospace; }
    .q-up { color: #10B981 !important; } .q-down { color: #EF4444 !important; } .q-neu { color: #F59E0B !important; }  
    .tag { padding: 3px 8px; border-radius: 2px; font-size: 0.75rem; font-weight: 700; margin-left: 8px; border: 1px solid; }
    .tag-moat { background-color: rgba(16, 185, 129, 0.1) !important; color: #10B981 !important; border-color: #10B981 !important; }
    .tag-risk { background-color: rgba(239, 68, 68, 0.1) !important; color: #EF4444 !important; border-color: #EF4444 !important; }
    .tag-logic { background-color: rgba(59, 130, 246, 0.1) !important; color: #3B82F6 !important; border-color: #3B82F6 !important; }
    .cmd-box { background-color: #1E293B !important; border: 1px solid #334155 !important; border-top: 4px solid #F59E0B !important; padding: 20px; border-radius: 4px; margin-bottom: 24px; }
    .stButton button, .stDownloadButton button { background-color: #1E293B !important; border: 1px solid #334155 !important; color: #F8FAFC !important; border-radius: 2px !important; }
    .stButton button:hover { border-color: #3B82F6 !important; color: #3B82F6 !important; background-color: #0D131F !important; }
    div.row-widget.stRadio > div { flex-direction: row; flex-wrap: wrap; gap: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 3. 初始化 ---
if 'raw_data' not in st.session_state: st.session_state['raw_data'] = None
if 'scan_finished' not in st.session_state: st.session_state['scan_finished'] = False
if 'history_storage' not in st.session_state: st.session_state['history_storage'] = {}
if 'ai_results' not in st.session_state: st.session_state['ai_results'] = {}
if 'current_logic' not in st.session_state: st.session_state['current_logic'] = "Quant" 
if 'panel_page' not in st.session_state: st.session_state['panel_page'] = 1 
if 'my_portfolio' not in st.session_state: st.session_state['my_portfolio'] = []

try: api_key = st.secrets["GEMINI_API_KEY"]
except: api_key = None

@st.cache_resource
def setup_chinese_font():
    font_path = "NotoSansTC-Regular.ttf"
    if not os.path.exists(font_path):
        try:
            req = urllib.request.Request("https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC-Regular.ttf", headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(font_path, 'wb') as out_file: out_file.write(response.read())
        except: pass
    try: pdfmetrics.registerFont(TTFont('ChineseFont', font_path)); return 'ChineseFont'
    except: return 'Helvetica'
font_name_global = setup_chinese_font()

def create_resilient_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session

@st.cache_data
def get_tw_stock_list():
    try:
        import twstock
        stock_map, industry_map = {}, {}
        for code, info in twstock.codes.items():
            if info.type in ['股票', 'ETF']:
                full = f"{code}.TW" if info.market == '上市' else f"{code}.TWO"
                stock_map[full] = f"{full} {info.name}"
                group = info.group if info.group else info.type
                if group not in industry_map: industry_map[group] = []
                industry_map[group].append(full)
        return stock_map, industry_map
    except: return {}, {}
stock_map, industry_map = get_tw_stock_list()

def get_stock_data(symbol):
    try:
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'): symbol += '.TW'
        session = create_resilient_session()
        ticker = yf.Ticker(symbol, session=session)
        try: info = ticker.info 
        except: info = {}
        try: hist = ticker.history(period="6mo")
        except: hist = pd.DataFrame()

        def g(k): return info.get(k)
        price = g('currentPrice') or g('previousClose')
        if (price is None or pd.isna(price)) and not hist.empty: price = float(hist['Close'].iloc[-1])
            
        return {
            'close_price': price, 'pe': g('trailingPE'), 'peg': g('pegRatio'), 'pb': g('priceToBook'),
            'rev_growth': g('revenueGrowth'), 'eps_growth': g('earningsGrowth'), 'trailing_eps': g('trailingEps'), 
            'gross_margins': g('grossMargins'), 'yield': g('dividendYield'), 'roe': g('returnOnEquity'),
            'beta': g('beta'), 'market_cap': g('marketCap'), 'fcf': g('freeCashflow'), 'debt_to_equity': g('debtToEquity'),
            'history': hist
        }
    except: return None

def calculate_implied_growth(price, eps, r=0.10, terminal_g=0.02, years=10):
    if pd.isna(price) or pd.isna(eps) or eps <= 0 or price <= 0: return np.nan
    low, high = -0.99, 3.0 
    for _ in range(50): 
        mid = (low + high) / 2
        calc_price = sum([eps * ((1 + mid) ** t) / ((1 + r) ** t) for t in range(1, years + 1)]) + ((eps * ((1 + mid) ** years) * (1 + terminal_g)) / (r - terminal_g)) / ((1 + r) ** years)
        if abs(calc_price - price) < 0.01: return mid
        if calc_price - price > 0: high = mid
        else: low = mid
    return (low + high) / 2

@st.cache_data(ttl=3600, show_spinner=False)
def batch_scan_stocks(stock_list):
    results, history_map = [], {}
    RISK_FREE_RATE = 0.015 
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(get_stock_data, s.split(' ')[0]): s for s in stock_list}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_str = future_to_stock[future]
            try:
                code = stock_str.split(' ')[0].split('.')[0]
                name = code 
                if len(stock_str.split(' ')) > 1: name = stock_str.split(' ')[1]
                
                y_data = future.result()
                if y_data is None or pd.isna(y_data.get('close_price')): continue

                price = y_data['close_price']
                volatility = sharpe = mdd = ma_bias = implied_g = np.nan
                
                hist = y_data.get('history')
                if hist is not None and not hist.empty:
                    history_map[code] = hist 
                    closes = hist['Close']
                    if len(closes) > 10:
                        volatility = closes.pct_change().std() * (252**0.5)
                        ma60 = closes.rolling(60).mean().iloc[-1]
                        if not pd.isna(ma60): ma_bias = (price / ma60) - 1
                        ann_ret = ((closes.iloc[-1] / closes.iloc[0]) - 1) * 2 
                        if volatility > 0: sharpe = (ann_ret - RISK_FREE_RATE) / volatility
                        mdd = (closes / closes.cummax() - 1.0).min()
                
                t_eps = y_data.get('trailing_eps')
                if pd.isna(t_eps) and y_data.get('pe'): t_eps = price / y_data.get('pe')
                implied_g = calculate_implied_growth(price, t_eps)

                ind = 'General'
                for k, v in industry_map.items():
                    if any(code in s for s in v): ind = k; break
                if code.startswith('00'): ind = 'ETF'

                results.append({
                    '代號': code, '名稱': name, 'close_price': price,
                    'pe': y_data.get('pe'), 'yield': (y_data.get('yield') or 0)*100 if y_data.get('yield') else np.nan, 
                    'roe': y_data.get('roe'), 'rev_growth': (y_data.get('rev_growth') or 0)*100 if y_data.get('rev_growth') else np.nan, 
                    'eps_growth': (y_data.get('eps_growth') or 0)*100 if y_data.get('eps_growth') else np.nan, 
                    'gross_margins': (y_data.get('gross_margins') or 0)*100 if y_data.get('gross_margins') else np.nan,
                    'fcf_yield': (y_data.get('fcf') / y_data.get('market_cap') * 100) if y_data.get('fcf') and y_data.get('market_cap') else np.nan, 
                    'de_ratio': y_data.get('debt_to_equity'), 'beta': y_data.get('beta'),
                    'sharpe': sharpe, 'mdd': mdd, 'implied_growth': implied_g, 'peg': y_data.get('peg'), 
                    'volatility': volatility, 'priceToMA60': ma_bias, 'industry': ind
                })
            except: continue
    df = pd.DataFrame(results)
    return df, history_map

def calculate_score(df, logic_type="Quant"):
    if df.empty: return df, None
    for col in ['pe', 'yield', 'rev_growth', 'eps_growth', 'gross_margins', 'fcf_yield', 'de_ratio', 'beta', 'sharpe', 'mdd', 'implied_growth', 'peg', 'volatility', 'roe', 'priceToMA60']:
        if col not in df.columns: df[col] = np.nan
    
    df_norm = df.copy()
    scores, plans, quality_tags = [], [], []
    calc_df = df.fillna({'pe': 50, 'volatility': 0.5, 'beta': 1.0, 'de_ratio': 100, 'sharpe':0, 'mdd':0})

    for idx, row in calc_df.iterrows():
        total_score = total_weight = 0
        if logic_type == "Quant": config = {'Sharpe': {'col': 'sharpe', 'dir': 'max', 'w': 4.0, 'cat': '動能'}, 'Volatility': {'col': 'volatility', 'dir': 'min', 'w': 2.0, 'cat': '風險'}, 'Beta': {'col': 'beta', 'dir': 'mid', 'w': 1.0, 'cat': '風險'}}
        else: config = {'ROE': {'col': 'roe', 'dir': 'max', 'w': 2.0, 'cat': '財報'}, 'GrossMargin': {'col': 'gross_margins', 'dir': 'max', 'w': 1.5, 'cat': '財報'}, 'FCF_Yield': {'col': 'fcf_yield', 'dir': 'max', 'w': 2.0, 'cat': '價值'}, 'DE_Ratio': {'col': 'de_ratio', 'dir': 'min', 'w': 1.5, 'cat': '風險'}, 'EPS_Growth': {'col': 'eps_growth', 'dir': 'max', 'w': 1.0, 'cat': '成長'}}
            
        for name, setting in config.items():
            rank = calc_df[setting['col']].rank(pct=True).get(idx, 0.5)
            norm = rank if setting['dir'] == 'max' else (1 - rank if setting['dir'] == 'min' else 1 - abs(rank - 0.5)*2)
            df_norm.loc[idx, f'{setting["cat"]}_n'] = norm * 100
            total_score += norm * 100 * setting['w']
            total_weight += setting['w']
            
        final = total_score / total_weight if total_weight > 0 else 50
        scores.append(round(final, 1))
        
        sh, vol, b, mdd = row.get('sharpe', 0), row.get('volatility', 1), row.get('beta', 1.0), row.get('mdd', 0)
        roe, gm, de, fcf = row.get('roe', 0), row.get('gross_margins', 0), row.get('de_ratio', 100), row.get('fcf_yield', 0)
        if pd.isna(b): b = 1.0
        
        q_tag = ""
        if logic_type == "Quant":
            if mdd < -0.25: q_tag = "回撤過大 (剔除)"; plans.append("跌破 25% 防線"); scores[-1] = 0 
            elif sh > 1.0 and b >= 1.0: q_tag = "成長型資產"; plans.append("配置主動能")
            elif sh > 1.0 and b < 1.0: q_tag = "防禦型資產"; plans.append("配置穩定基石")
            elif sh > 0 and vol < 0.25: q_tag = "中性資產"; plans.append("降低組合波動")
            elif sh < 0: q_tag = "風險報酬不對等"; plans.append("避開標的")
            else: q_tag = "中立觀望"; plans.append("中立觀望")
        else: 
            if mdd < -0.25: q_tag = "回撤過大 (剔除)"; plans.append("跌破 25% 防線"); scores[-1] = 0
            elif roe > 15 and gm > 40 and de < 100 and fcf > 0: q_tag = "護城河優良"; plans.append("價值浮現")
            elif de > 200 or fcf < -5: q_tag = "財務風險"; plans.append("避開標的")
            else: q_tag = "中立觀望"; plans.append("合理估值" if final > 60 else "中立觀望")
            
        quality_tags.append(q_tag)
            
    df['Score'], df['Strategy'], df['Quality'] = scores, plans, quality_tags
    return df.sort_values('Score', ascending=False), df_norm

def greedy_mpt_optimization(pool_df, returns_df, target_n, metric_col, initial_selected=None, target_vol_max=0.15):
    selected_codes = list(initial_selected) if initial_selected else []
    candidates = [c for c in pool_df['代號'].tolist() if c not in selected_codes]
    if not candidates: return selected_codes
    max_metric, min_metric = pool_df[metric_col].max(), pool_df[metric_col].min()
    if max_metric == min_metric: max_metric = min_metric + 1 
    
    if not selected_codes:
        first_code = pool_df.loc[pool_df[metric_col].idxmax(), '代號']
        selected_codes.append(first_code)
        candidates.remove(first_code)
        
    while len(selected_codes) < (len(initial_selected or []) + target_n) and candidates:
        best_score, best_code = -999999, None
        for code in candidates:
            norm_metric = (pool_df[pool_df['代號'] == code][metric_col].values[0] - min_metric) / (max_metric - min_metric)
            corrs = [returns_df[code].corr(returns_df[sc]) for sc in selected_codes if code in returns_df.columns and sc in returns_df.columns]
            corrs = [c for c in corrs if not pd.isna(c)]
            corr_penalty = (np.mean(corrs) * 5.0 + max(0, max(corrs) - 0.4) * 10.0) if corrs else 0
            
            valid_codes = [c for c in selected_codes + [code] if c in returns_df.columns]
            port_vol = 0.20 
            if len(valid_codes) > 1:
                cov_matrix = returns_df[valid_codes].cov() * 252
                w = np.array([0.135 if pool_df[pool_df['代號']==tc]['戰略定位'].values[0]=="🚀 成長型資產" else 0.08 for tc in valid_codes])
                port_variance = np.dot((w/np.sum(w)).T, np.dot(cov_matrix, w/np.sum(w)))
                port_vol = np.sqrt(port_variance)
                
            vol_penalty = (port_vol - target_vol_max) * 100.0 if port_vol > target_vol_max else 0
            obj_score = norm_metric - corr_penalty - vol_penalty 
            if obj_score > best_score: best_score, best_code = obj_score, code
        if best_code: selected_codes.append(best_code); candidates.remove(best_code)
        else: break
    return selected_codes

def generate_custom_signal(row, regime):
    s, mdd, b, tag = row.get('sharpe', 0), row.get('mdd', 0), row.get('beta', 1.0), row.get('Quality', '')
    if mdd < -0.25: return "🛑 強制停損 (MDD破防)"
    if s < 0: return "⚠️ 劣勢資產 (建議剔除)"
    if regime == "Bull":
        if tag == "成長型資產" and s > 2.0 and mdd > -0.10: return "🔥 強勢加碼 (放大獲利)"
        if tag in ["防禦型資產", "中性資產"]: return "📉 逢高減碼 (釋放資金)"
        return "⚖️ 持有續抱"
    else: 
        if b > 1.2: return "📉 降持高波 (收回現金)"
        if tag in ["防禦型資產", "中性資產"]: return "🛡️ 逢低加碼 (鞏固底盤)"
        return "⚖️ 持有觀望"

# --- 15. 主介面 ---
with st.sidebar:
    st.title("⚙️ 終端控制面板")
    logic_choice_tw = st.radio("選擇運算引擎", ["📊 量化風控模型 (夏普值/波動率/Beta)", "👴 價值護城河 (高ROE/毛利/現金流)"], label_visibility="collapsed")
    st.session_state['current_logic'] = "Quant" if "量化" in logic_choice_tw else "Buffett"
    st.markdown("---")
    
    scan_mode = st.radio("篩選維度", ["市場焦點策略", "產業族群板塊", "台灣 ETF 專區", "自訂代碼輸入"])
    target_stocks = []
    if scan_mode == "自訂代碼輸入":
        target_stocks = st.multiselect("搜尋上市櫃標的", list(stock_map.values()), default=["2330.TW 台積電", "2454.TW 聯發科"])
        if manual := st.text_input("快速輸入代號 (如 2317)"): target_stocks.append(f"{manual}.TW")
    elif scan_mode == "產業族群板塊":
        selected_inds = st.multiselect("板塊選擇", ["[ALL] 載入全部板塊"] + sorted(list(industry_map.keys())))
        for k in (industry_map.keys() if "[ALL] 載入全部板塊" in selected_inds else selected_inds): target_stocks.extend(industry_map[k])
    elif scan_mode == "台灣 ETF 專區":
        target_stocks.extend(["0050.TW", "0056.TW", "00878.TW", "00881.TW", "00919.TW", "00929.TW"]) # 簡化示範
    else: 
        target_stocks.extend(["2382.TW", "3231.TW", "2356.TW", "2376.TW", "2308.TW", "2330.TW", "2317.TW", "3017.TW", "3324.TW"]) # 預設加入AI代工與台達電
        st.info("💡 預設已載入「AI伺服器與零組件」強勢板塊 (含廣達、緯創、台達電等)")

    target_stocks = list(dict.fromkeys(target_stocks)) 

    if st.button("🚀 啟動終端運算", type="primary", use_container_width=True):
        st.session_state['scan_finished'] = False
        with st.spinner(f"正在擷取並運算 {len(target_stocks)} 檔標的數據..."):
            raw, hist_map = batch_scan_stocks(target_stocks)
            st.session_state['raw_data'], st.session_state['history_storage'], st.session_state['scan_finished'] = raw, hist_map, True
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 台股量化與價值分析終端")
    st.caption(f"STATUS: ONLINE | ENGINE: **{'量化風控' if st.session_state['current_logic'] == 'Quant' else '價值護城河'}** | ALGO: 動態權重與 MDD 絕對防禦機制")

if st.session_state['scan_finished'] and st.session_state['raw_data'] is not None:
    final_df, df_norm = calculate_score(st.session_state['raw_data'], logic_type=st.session_state['current_logic'])
    
    st.subheader("🏆 終端檢索清單")
    tags = sorted(final_df['Quality'].dropna().unique().tolist(), key=lambda t: 0 if "成長" in t else (1 if "防禦" in t else (2 if "中性" in t else 3)))
    selected_filter = st.radio("🎯 快速分類篩選：", ["[ALL] 顯示全部標的"] + tags, horizontal=True)
    filtered_df = final_df if selected_filter == "[ALL] 顯示全部標的" else final_df[final_df['Quality'] == selected_filter]
    
    df_event = st.dataframe(
        filtered_df[['代號', '名稱', 'industry', 'Score', 'Quality', 'sharpe', 'mdd', 'roe']],
        column_config={"Score": st.column_config.ProgressColumn("評分", format="%.1f"), "sharpe": st.column_config.NumberColumn("夏普值", format="%.2f"), "mdd": st.column_config.NumberColumn("最大回撤", format="%.1f%%")},
        use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row" 
    )
    
    if df_event.selection.rows:
        sel_row = filtered_df.iloc[df_event.selection.rows[0]]
        c1, c2 = st.columns([3, 1])
        c1.subheader(f"💡 深度解析：{sel_row['名稱']} ({sel_row['代號']})")
        if c2.button("❌ 從自選移除" if sel_row['代號'] in st.session_state['my_portfolio'] else "➕ 加入自選戰略組合", use_container_width=True):
            st.session_state['my_portfolio'].remove(sel_row['代號']) if sel_row['代號'] in st.session_state['my_portfolio'] else st.session_state['my_portfolio'].append(sel_row['代號'])
            st.rerun()

    # =========================================================
    # 🛠️ 戰略指揮中心：自選組合監控與多空情境推演
    # =========================================================
    st.markdown("---")
    st.subheader("🛠️ 戰略指揮中心：自選組合監控與多空情境推演")
    
    if not st.session_state['my_portfolio']:
        st.info("💡 請在上方清單點選標的，並點擊【➕ 加入自選戰略組合】以啟用監控與推演功能。")
    else:
        regime = st.radio("🌍 切換總經市場情境", ["📈 多頭市場 (Bull Market) - 放大獲利", "📉 空頭/震盪市場 (Bear Market) - 著重防禦"], horizontal=True)
        my_port_df = final_df[final_df['代號'].isin(st.session_state['my_portfolio'])].copy()
        
        my_port_df['資產屬性'] = my_port_df.apply(lambda r: "🛑 剔除資產" if r.get('mdd',0) < -0.25 else ("🚀 成長型資產" if r.get('sharpe',0)>1 and r.get('beta',1)>=1 else ("🛡️ 防禦型資產" if r.get('sharpe',0)>1 and r.get('beta',1)<1 else ("⚖️ 中性資產" if r.get('sharpe',0)>0 and r.get('volatility',1)<0.25 else "🟡 觀察資產"))), axis=1)
        my_port_df['操作訊號'] = my_port_df.apply(lambda r: generate_custom_signal(r, "Bull" if "多頭" in regime else "Bear"), axis=1)
        
        st.dataframe(my_port_df[['代號', '名稱', '資產屬性', 'sharpe', 'beta', 'mdd', '操作訊號']], hide_index=True, use_container_width=True)
        
        counts = my_port_df['資產屬性'].value_counts()
        if "多頭" in regime: t_grow, t_neu, t_def, t_cash, adv = 60, 25, 10, 5, "提高成長型資產配置至 60%，放大高 Sharpe 動能標的權重。"
        else: t_grow, t_neu, t_def, t_cash, adv = 30, 30, 25, 15, "防禦優先，降低高 Beta 持倉，將防禦與中性資產拉高至 55%，並保留充裕現金。"
        
        st.markdown(f"<div class='cmd-box'><b>📝 情境推演建議：</b><br>{adv}</div>", unsafe_allow_html=True)
        w_c1, w_c2, w_c3, w_c4 = st.columns(4)
        for col, title, tgt, count in zip([w_c1, w_c2, w_c3, w_c4], ["🚀 成長型資產", "⚖️ 中性資產", "🛡️ 防禦型資產", "💵 現金部位"], [t_grow, t_neu, t_def, t_cash], [counts.get("🚀 成長型資產",0), counts.get("⚖️ 中性資產",0), counts.get("🛡️ 防禦型資產",0), "-"]):
            col.markdown(f"<div style='background-color:#06090F; padding:15px; border-radius:4px; border:1px solid #1E293B;'><div style='color:#9CA3AF;'>{title}</div><div style='font-size:1.5rem; font-weight:bold; color:#F8FAFC;'>目標 {tgt}%</div><div style='color:#3B82F6;'>目前自選: {count}</div></div>", unsafe_allow_html=True)

    # =========================================================
    # 💼 系統嚴選：自動化 10 檔模型專屬投資組合
    # =========================================================
    st.markdown("---")
    st.subheader("💼 系統嚴選：模型專屬效率前緣配置")
    
    returns_df = pd.concat([h['Close'].pct_change().rename(c) for c, h in hist_storage.items() if not h.empty], axis=1).dropna(how='all').fillna(0) if hist_storage else pd.DataFrame()
    
    if st.session_state['current_logic'] == "Quant":
        pool_df = final_df[(final_df['sharpe'] > 0) & (final_df['mdd'] >= -0.25)].copy()
        if len(pool_df)==0: pool_df = final_df.head(10).copy()
        pool_df['戰略定位'] = pool_df.apply(lambda r: "🚀 成長型資產" if r.get('sharpe',0)>1 and r.get('beta',1)>=1 else ("🛡️ 防禦型資產" if r.get('sharpe',0)>1 and r.get('beta',1)<1 else "⚖️ 中性資產"), axis=1)
        
        final_codes = greedy_mpt_optimization(pool_df[pool_df['戰略定位']=="🚀 成長型資產"], returns_df, 4, 'sharpe', target_vol_max=0.15)
        final_codes = greedy_mpt_optimization(pool_df[pool_df['戰略定位']=="🛡️ 防禦型資產"], returns_df, 3, 'sharpe', initial_selected=final_codes)
        final_codes = greedy_mpt_optimization(pool_df[pool_df['戰略定位']=="⚖️ 中性資產"], returns_df, 3, 'sharpe', initial_selected=final_codes)
        if len(final_codes)<10: final_codes.extend(greedy_mpt_optimization(pool_df[~pool_df['代號'].isin(final_codes)], returns_df, 10-len(final_codes), 'sharpe', initial_selected=final_codes))
        
        port_df = pool_df[pool_df['代號'].isin(final_codes)]
        port_df['建議權重'] = port_df.apply(lambda r: "12%~15%" if r['戰略定位']=="🚀 成長型資產" and r['sharpe']>2 and r['volatility']<0.3 else ("8%~12%" if r['戰略定位'] in ["🚀 成長型資產","🛡️ 防禦型資產"] else "5%~8%"), axis=1)
        st.dataframe(port_df[['代號', '名稱', '戰略定位', 'sharpe', 'mdd', '建議權重']], hide_index=True, use_container_width=True)
