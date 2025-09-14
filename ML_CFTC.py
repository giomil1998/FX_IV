import pandas as pd
import yfinance as yf
import glob
import os
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
users = ["Lev_Money", "Dealer", "Asset_Mgr", "Other_Rept", "Tot_Rept", "NonRept"]
currency = "AUSTRALIAN"
ticker = "AUDUSD=X"
rsi_horizon = 4
short_ma = 4
long_ma = 52
momentum_lookback = 4
fwd_return_window = 3
folder_path = "/Users/giorgiomilitello/Desktop/Extension project/CFTC/Financial Futures/"

# --- Load CFTC Data ---
file_pattern = os.path.join(folder_path, "FinFutYY_*.xls")
excel_files = glob.glob(file_pattern)
df_list = [pd.read_excel(file) for file in excel_files]
full_df = pd.concat(df_list, ignore_index=True)

# --- Filter for currency ---
curr_df = full_df[full_df["Market_and_Exchange_Names"].str.contains(currency, case=False)].copy()
curr_df['Report_Date_as_MM_DD_YYYY'] = pd.to_datetime(curr_df['Report_Date_as_MM_DD_YYYY'])
curr_df.set_index('Report_Date_as_MM_DD_YYYY', inplace=True)
curr_df.sort_index(inplace=True)

# --- Download spot FX data ---
fx_series = yf.download(ticker, start=curr_df.index.min().strftime('%Y-%m-%d'),
                        end=curr_df.index.max().strftime('%Y-%m-%d'))['Close']
fx_series = fx_series.resample('W-TUE').last()

# --- Download bond ETF prices ---
bond_tickers = {"GOVT": "US_Bond"}
bond_series = yf.download(list(bond_tickers.keys()),
                          start=curr_df.index.min().strftime('%Y-%m-%d'),
                          end=curr_df.index.max().strftime('%Y-%m-%d'))["Close"]
bond_series.columns = [bond_tickers[t] for t in bond_series.columns]
bond_series = bond_series.resample('W-TUE').last()

# --- Merge and forward-fill ---
combined = pd.merge(fx_series, curr_df, left_index=True, right_index=True, how='left')
combined = combined.sort_index().ffill()
combined = combined.join(bond_series, how='left')
combined['US_Bond'] = combined['US_Bond'].ffill()

# --- Add bond momentum features ---
combined['US_Bond_mom'] = combined['US_Bond'].pct_change(periods=momentum_lookback)

# --- Add RSI and Momentum ---
rsi = RSIIndicator(close=combined[ticker], window=rsi_horizon)
combined['rsi'] = rsi.rsi()
combined['momentum'] = combined[ticker].pct_change(periods=momentum_lookback)

# --- Feature Engineering ---
for user in users:
    net = f'Net_{user}'
    combined[net] = combined[f'{user}_Positions_Long_All'] - combined[f'{user}_Positions_Short_All']
    combined[f'z_{net}'] = (combined[net] - combined[net].rolling(long_ma).mean()) / combined[net].rolling(long_ma).std()
    combined[f'{user}_Long_OI_Change'] = combined[f'Pct_of_OI_{user}_Long_All'].diff()
    combined[f'{user}_Short_OI_Change'] = combined[f'Pct_of_OI_{user}_Short_All'].diff()

# --- Target Variable (Buy/Sell) ---
# For 3-week forward return
combined['return'] = combined[ticker].pct_change(periods=fwd_return_window).shift(-fwd_return_window)
def classify_return(r):
    if r > 0.001:
        return 1
    elif r < -0.001:
        return -1
    else:
        return 0

combined['target'] = combined['return'].apply(classify_return)

# --- Drop NaNs ---
combined = combined.dropna()

# --- Align features to available CFTC date (1-week delay) ---
features = combined[[
    f'Net_{u}' for u in users
] + [
    f'z_Net_{u}' for u in users
] + [
    f'{u}_Long_OI_Change' for u in users
] + [
    f'{u}_Short_OI_Change' for u in users
] + ['rsi', 'momentum',  'US_Bond_mom']]
features = features.shift(1)

# --- Align and drop final NaNs ---
data = pd.concat([features, combined[[ticker, 'return', 'target']]], axis=1).dropna()
print("\n--- DEBUG INFO BEFORE MODEL TRAINING ---")
print("combined shape:", combined.shape)
print("features shape:", features.shape)
print("data shape after concat + dropna:", data.shape)
print("Number of available samples:", len(data))
print("Date range in data:", data.index.min(), "to", data.index.max())
X = data.drop(columns=['target', ticker, 'return'])
y = data['target']

# --- Train/Test Split ---
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    data['ml_signal'] = model.predict(X)

    plot_dir = f"/Users/giorgiomilitello/Desktop/Extension project/CFTC/Plots/ML"
    os.makedirs(plot_dir, exist_ok=True)

    full_fx_series = yf.download(ticker, start=curr_df.index.min().strftime('%Y-%m-%d'),
                                 end=curr_df.index.max().strftime('%Y-%m-%d'))['Close']

    plt.figure(figsize=(14, 6))
    plt.plot(full_fx_series.index, full_fx_series, label='Daily Spot Price', color='blue')
    plt.scatter(data[data['ml_signal'] == 1].index, data[data['ml_signal'] == 1][ticker],
                label='Buy Signal (weekly)', color='green', marker='^', zorder=5)
    plt.scatter(data[data['ml_signal'] == -1].index, data[data['ml_signal'] == -1][ticker],
                label='Sell Signal (weekly)', color='red', marker='v', zorder=5)
    plt.title(f"{currency} Spot Price and ML-Based Signals")
    plt.xlabel("Date")
    plt.ylabel("Spot Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{currency}_ml_signals.png"))
    plt.show()

    importances = model.feature_importances_
    feat_names = X.columns
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [feat_names[i] for i in sorted_idx])
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{currency}_ml_feature_importance.png"))
    plt.show()

    data['next_return'] = data['return'].shift(-1)
    plt.figure(figsize=(10, 4))
    plt.hist(data[data['ml_signal'] == 1]['next_return'], bins=30, alpha=0.6, label='Buy Signal')
    plt.hist(data[data['ml_signal'] == -1]['next_return'], bins=30, alpha=0.6, label='Sell Signal')
    plt.legend()
    plt.title("Next Week Return Distribution by ML Signal")
    plt.tight_layout()
    plt.show()

    data['next_price'] = data[ticker].shift(-1)
    data = data.dropna(subset=['next_price'])

    test_data = data.loc[X_test.index.intersection(data.index)]
    signal_mask = test_data['ml_signal'].isin([1, -1])
    signal_sign = test_data.loc[signal_mask, 'ml_signal']
    print(signal_sign)
    signal_returns = test_data.loc[signal_mask, 'next_price'] - test_data.loc[signal_mask, ticker]

    wins = ((signal_sign == 1) & (signal_returns > 0)) | ((signal_sign == -1) & (signal_returns < 0))
    win_rate = wins.mean()
    # if not wins.empty else 0.0
    print(f"\nOut-of-sample win rate (buy/sell only): {win_rate:.2%} on {len(wins)} trades")
    print("\n--- DEBUG INFO ---")
    print("X_test shape:", X_test.shape)
    print("y_pred shape:", y_pred.shape)
    print("test_data shape:", test_data.shape)
    print("signal_mask shape:", signal_mask.shape)
    print("signal_mask sum (number of signals):", signal_mask.sum())
    print("signal_sign shape:", signal_sign.shape)
    print("signal_returns shape:", signal_returns.shape)
    print("wins shape:", wins.shape)
else:
    print("Not enough data to run model training and evaluation.")