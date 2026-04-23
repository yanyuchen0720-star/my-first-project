import requests
import pandas as pd
import matplotlib.pyplot as plt
import urllib3
import time
import sys
import numpy as np
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 模組 1：從證交所抓取【價量資料】
# ==========================================
def fetch_history_data(stock_id, end_date_obj, months_count=5):
    all_data = []
    headers = {"User-Agent": "Mozilla/5.0"}
    y, m = end_date_obj.year, end_date_obj.month
    
    for i in range(months_count):
        target_date = f"{y}{m:02d}01"
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={target_date}&stockNo={stock_id}"
        print(f"🔄 [證交所] 正在抓取 {target_date[:6]} 的股價資料...")
        try:
            res = requests.get(url, headers=headers, verify=False).json()
            if res.get('stat') == 'OK':
                all_data.append(pd.DataFrame(res['data'], columns=res['fields']))
            elif i == 0: 
                sys.exit(f"\n❌ 找不到股票代號【{stock_id}】！")
            else: 
                break
        except Exception as e:
            print(f"連線錯誤: {e}"); break
            
        m -= 1
        if m == 0: m = 12; y -= 1
        time.sleep(3) 
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

# ==========================================
# 模組 2：從 FinMind 抓取【三大法人】
# ==========================================
def fetch_institutional_data(stock_id, end_date_obj, months_count=5):
    print("🔄 [FinMind] 正在獲取三大法人籌碼資料...")
    start_date = (end_date_obj - pd.Timedelta(days=30 * months_count)).strftime("%Y-%m-%d")
    end_date = end_date_obj.strftime("%Y-%m-%d")
    
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date
    }
    
    # 🔑 你的 Token 已寫入
    finmind_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoieWFueXVDaGVuMDcyMCIsImVtYWlsIjoieWFueXVjaGVuMDcyMEBnbWFpbC5jb20ifQ.7uyKc8PxHdqa9ENaoYF4RkNg4TWznPFQhiPE4bXB670" 
    if finmind_token:
        params["token"] = finmind_token
    
    try:
        res = requests.get(url, params=params, verify=False).json()
        
        if res.get('msg') != 'success':
            print(f"⚠️ FinMind API 拒絕請求: {res.get('msg')}")
            return None
            
        if not res.get('data'):
            print(f"\n⚠️ 警告：FinMind 找不到 {start_date} 到 {end_date} 的資料！")
            return None
            
        df = pd.DataFrame(res['data'])
        df['net_buy'] = (df['buy'] - df['sell']) / 1000
        
        def group_name(name):
            name_lower = str(name).lower()
            if 'foreign' in name_lower or '外資' in name_lower: return 'Foreign'
            if 'trust' in name_lower or '投信' in name_lower: return 'Trust'
            if 'dealer' in name_lower or '自營' in name_lower: return 'Dealer'
            return 'Other'
            
        df['type'] = df['name'].apply(group_name)
        pivot_df = df.pivot_table(index='date', columns='type', values='net_buy', aggfunc='sum').fillna(0)
        pivot_df.index = pd.to_datetime(pivot_df.index)
        return pivot_df
        
    except Exception as e:
        print(f"三大法人抓取異常: {e}")
    return None

# ==========================================
# 模組 3：指標計算與資料清洗
# ==========================================
def process_data(df):
    df = df[['日期', '成交股數', '開盤價', '最高價', '最低價', '收盤價']].copy()
    for col in ['成交股數', '開盤價', '最高價', '最低價', '收盤價']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.ffill()
    
    df['日期'] = pd.to_datetime(df['日期'].apply(lambda d: f"{int(d.split('/')[0])+1911}/{d.split('/')[1]}/{d.split('/')[2]}"))
    df = df.sort_values('日期').reset_index(drop=True)
    
    # 🌟 補回缺失的均線、EMA、布林通道計算
    df['MA10'] = df['收盤價'].rolling(10).mean()
    df['MA20'] = df['收盤價'].rolling(20).mean()
    
    df['EMA10'] = df['收盤價'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['收盤價'].ewm(span=20, adjust=False).mean()
    
    df['BB_MID'] = df['MA20']
    std20 = df['收盤價'].rolling(window=20).std()
    df['BB_UP'] = df['BB_MID'] + 2 * std20
    df['BB_LOW'] = df['BB_MID'] - 2 * std20
    
    df['Volume'] = df['成交股數'] / 1000
    
    # KD
    df['9H'] = df['最高價'].rolling(window=9, min_periods=1).max()
    df['9L'] = df['最低價'].rolling(window=9, min_periods=1).min()
    denominator = df['9H'] - df['9L']
    df['RSV'] = np.where(denominator == 0, 50, 100 * (df['收盤價'] - df['9L']) / denominator)
    df['RSV'] = pd.Series(df['RSV']).fillna(50)
    
    K_list, D_list = [], []
    k_prev, d_prev = 50, 50
    for rsv in df['RSV']:
        k_curr = (2/3) * k_prev + (1/3) * rsv
        d_curr = (2/3) * d_prev + (1/3) * k_curr
        K_list.append(k_curr); D_list.append(d_curr)
        k_prev, d_prev = k_curr, d_curr
    df['K'], df['D'] = K_list, D_list
    
    # MACD
    df['MACD'] = df['收盤價'].ewm(span=12, adjust=False).mean() - df['收盤價'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    # RSI
    delta = df['收盤價'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=6, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=6, min_periods=1).mean()
    df['RSI6'] = 100 - (100 / (1 + (gain / loss)))
    
    # WR
    df['WR'] = -100 * (df['最高價'].rolling(14).max() - df['收盤價']) / (df['最高價'].rolling(14).max() - df['最低價'].rolling(14).min())
    
    # CCI
    tp = (df['最高價'] + df['最低價'] + df['收盤價']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean()))
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['收盤價'].iloc[i] > df['收盤價'].iloc[i-1]: obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['收盤價'].iloc[i] < df['收盤價'].iloc[i-1]: obv.append(obv[-1] - df['Volume'].iloc[i])
        else: obv.append(obv[-1])
    df['OBV'] = obv
    
    return df

# ==========================================
# 模組 4：副圖繪製器
# ==========================================
def draw_sub_indicator(ax, x_pos, df, ind_type, future_len):
    hist_x = x_pos[:-future_len] if future_len > 0 else x_pos
    ind_type = ind_type.upper()
    
    if ind_type == 'VOL':
        ax.bar(hist_x, df['Volume'], color='teal', alpha=0.6); ax.set_ylabel("VOL")
    elif ind_type == 'KD':
        ax.plot(hist_x, df['K'], label='K', color='darkorange')
        ax.plot(hist_x, df['D'], label='D', color='royalblue')
        ax.axhline(80, color='red', linestyle='--', alpha=0.4)
        ax.axhline(20, color='green', linestyle='--', alpha=0.4)
        ax.set_ylim(0, 100); ax.set_ylabel("KD"); ax.legend(loc='upper left', fontsize=8)
    elif ind_type == 'MACD':
        colors = ['red' if h > 0 else 'green' for h in df['Hist']]
        ax.bar(hist_x, df['Hist'], color=colors, alpha=0.5)
        ax.plot(hist_x, df['MACD'], label='MACD', color='blue')
        ax.plot(hist_x, df['Signal'], label='Sig', color='orange')
        ax.set_ylabel("MACD"); ax.legend(loc='upper left', fontsize=8)
    elif ind_type == 'RSI':
        ax.plot(hist_x, df['RSI6'], label='RSI(6)', color='purple')
        ax.axhline(80, color='red', linestyle='--', alpha=0.3); ax.axhline(20, color='green', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 100); ax.set_ylabel("RSI"); ax.legend(loc='upper left', fontsize=8)
    elif ind_type == 'WR':
        ax.plot(hist_x, df['WR'], label='WR(14)', color='magenta')
        ax.axhline(-20, color='red', linestyle='--', alpha=0.3); ax.axhline(-80, color='green', linestyle='--', alpha=0.3)
        ax.set_ylim(-100, 0); ax.set_ylabel("WR"); ax.legend(loc='upper left', fontsize=8)
    elif ind_type == 'CCI':
        ax.plot(hist_x, df['CCI'], label='CCI(20)', color='brown')
        ax.axhline(100, color='red', linestyle='--', alpha=0.3); ax.axhline(-100, color='green', linestyle='--', alpha=0.3)
        ax.set_ylabel("CCI"); ax.legend(loc='upper left', fontsize=8)
    elif ind_type == 'OBV':
        ax.plot(hist_x, df['OBV'], label='OBV', color='navy'); ax.set_ylabel("OBV"); ax.legend(loc='upper left', fontsize=8)
    elif ind_type == 'INST':
        if 'Total_Inst' not in df.columns: return
        colors = ['red' if val > 0 else 'green' for val in df['Total_Inst']]
        ax.bar(hist_x, df['Total_Inst'], color=colors, alpha=0.3, label='Total Net')
        ax.plot(hist_x, df['Foreign'], color='blue', label='Foreign', linewidth=1.5)
        ax.plot(hist_x, df['Trust'], color='darkorange', label='Trust', linewidth=1.5)
        ax.plot(hist_x, df['Dealer'], color='purple', label='Dealer', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel("Inst. Net"); ax.legend(loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, f"Unknown: {ind_type}", ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel("Error")
        
    ax.grid(True, linestyle='--', alpha=0.5)

# ==========================================
# 模組 5：主程式引擎
# ==========================================
if __name__ == "__main__":
    stock_no = input("🔍 輸入股票代號 (例如 2330): ").strip()
    if not stock_no: sys.exit()
        
    # 🌟 完全自由的日期輸入 (無強制校準)
    date_input = input("📅 輸入基準日 (YYYYMMDD，按 Enter 預設為系統今日): ").strip()
    target_date = datetime.now() if not date_input else datetime.strptime(date_input, "%Y%m%d")

    print(f"\n🚀 開始分析 {stock_no}，實際基準日將使用：{target_date.strftime('%Y-%m-%d')} ...\n")

    print("\n" + "="*40)
    print("📈 可用主圖指標：MA (一般均線), EMA (指數均線), BB (布林通道)")
    main_ind = input("👉 請輸入主圖指標代號 (預設 MA): ").strip().upper() or "MA"

    print("\n📉 可用副圖指標：VOL, KD, MACD, RSI, WR, CCI, OBV, INST (三大法人籌碼)")
    sub_input = input("👉 請輸入副圖指標代號 (用逗號隔開，例如 KD,INST): ").strip().upper().replace('，', ',')
    sub_inds = [i.strip() for i in sub_input.split(',')] if sub_input else ["VOL", "INST"]
    print("="*40 + "\n")

    # 抓取資料
    raw_df = fetch_history_data(stock_no, target_date)
    if raw_df is None: sys.exit("❌ 無法取得證交所資料。")
    full_df = process_data(raw_df)
    
    inst_df = fetch_institutional_data(stock_no, target_date)
    if inst_df is not None:
        full_df = pd.merge(full_df, inst_df, left_on='日期', right_index=True, how='left').fillna(0)
    for col in ['Foreign', 'Trust', 'Dealer']:
        if col not in full_df.columns: full_df[col] = 0
    full_df['Total_Inst'] = full_df['Foreign'] + full_df['Trust'] + full_df['Dealer']
    
    # 切割近期 60 天
    plot_df = full_df[full_df['日期'] <= target_date].tail(60).reset_index(drop=True)
    if plot_df.empty: sys.exit("❌ 該日期區間無資料。")

    # ==========================================
    # 模組 6：蒙地卡羅模擬 (預測引擎)
    # ==========================================
    forecast_days = 30
    
    returns = plot_df['收盤價'].pct_change().dropna().values
    mu = np.mean(returns)      # 期望日報酬
    sigma = np.std(returns)    # 日波動率
    last_price = plot_df['收盤價'].iloc[-1]
    
    x_future = np.arange(len(plot_df) - 1, len(plot_df) + forecast_days)
    
    num_simulations = 1000
    simulations = np.zeros((forecast_days + 1, num_simulations))
    simulations[0] = last_price
    
    for t in range(1, forecast_days + 1):
        random_shocks = np.random.normal(0, 1, num_simulations)
        simulations[t] = simulations[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * random_shocks)
        
    expected_path = np.mean(simulations, axis=1)
    upper_bound = np.percentile(simulations, 95, axis=1)
    lower_bound = np.percentile(simulations, 5, axis=1)
    
    last_date = plot_df['日期'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    all_x = np.concatenate([np.arange(len(plot_df)), x_future[1:]])
    all_dates = pd.concat([plot_df['日期'], pd.Series(future_dates)])
    
    # ==========================================
    # 繪圖區塊
    # ==========================================
    num_subs = len(sub_inds)
    total_plots = 1 + num_subs
    fig_height = 6 + (2.5 * num_subs)
    height_ratios = [3] + [1] * num_subs
    
    fig, axes = plt.subplots(total_plots, 1, figsize=(14, fig_height), gridspec_kw={'height_ratios': height_ratios}, sharex=True)
    fig.suptitle(f"Pro Analysis & MC Forecast ({stock_no}) - Base Date: {target_date.strftime('%Y-%m-%d')}", fontsize=16, fontweight='bold')
    
    if total_plots == 1: axes = [axes]
    ax_main = axes[0]
    x_hist = np.arange(len(plot_df))

    # 畫 K 棒
    for i in x_hist:
        r = plot_df.iloc[i]
        c = 'red' if r['收盤價'] > r['開盤價'] else 'green'
        ax_main.vlines(i, r['最低價'], r['最高價'], color='black', linewidth=1)
        ax_main.bar(i, abs(r['收盤價']-r['開盤價']), bottom=min(r['開盤價'], r['收盤價']), color=c, width=0.6, edgecolor='black')

    # 主指標 (包含 BB 與 EMA)
    if main_ind == 'MA' or main_ind not in ['EMA', 'BB']:
        ax_main.plot(x_hist, plot_df['MA10'], color='dodgerblue', label='10MA', alpha=0.7)
        ax_main.plot(x_hist, plot_df['MA20'], color='purple', label='20MA', alpha=0.7)
    elif main_ind == 'EMA':
        ax_main.plot(x_hist, plot_df['EMA10'], color='cyan', label='10EMA', alpha=0.7)
        ax_main.plot(x_hist, plot_df['EMA20'], color='magenta', label='20EMA', alpha=0.7)
    elif main_ind == 'BB':
        ax_main.plot(x_hist, plot_df['BB_MID'], color='blue', label='20MA(Mid)', linestyle='--')
        ax_main.plot(x_hist, plot_df['BB_UP'], color='gray', label='UP')
        ax_main.plot(x_hist, plot_df['BB_LOW'], color='gray', label='LOW')
        ax_main.fill_between(x_hist, plot_df['BB_LOW'], plot_df['BB_UP'], color='gray', alpha=0.1)

    # 繪製蒙地卡羅預測模型
    ax_main.plot(x_future, expected_path, color='magenta', linestyle='--', linewidth=2.5, label='MC Forecast (Mean)')
    ax_main.fill_between(x_future, lower_bound, upper_bound, color='magenta', alpha=0.15, label='95% Confidence')
    ax_main.axvline(x_hist[-1], color='red', linestyle=':', linewidth=2)
    ax_main.set_ylabel("Price")
    ax_main.legend(loc='upper left', fontsize=9)
    ax_main.grid(True, linestyle='--', alpha=0.5)

    # 畫副圖
    for i, ind in enumerate(sub_inds):
        if i + 1 < len(axes):
            draw_sub_indicator(axes[i+1], all_x, plot_df, ind, forecast_days)

    plt.xlim(all_x[0] - 0.5, all_x[-1] + 0.5)
    axes[-1].set_xticks(all_x[::3])
    axes[-1].set_xticklabels(all_dates.dt.strftime('%m-%d').tolist()[::3], rotation=45)
    
    plt.tight_layout()
    print("\n✅ 運算完成！圖表已彈出，請查看新視窗。")
    plt.show()
