import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

currency = 'USDNZD'

# Load the Excel file
file_path = f'/Users/giorgiomilitello/Desktop/Extension project/{currency}.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load the FOMC dates and MPS data
fomc_dates = pd.read_csv('/Users/giorgiomilitello/Desktop/Extension project/FOMCdates.csv')
fomc_dates['End'] = pd.to_datetime(fomc_dates['End'])

mps_path = '/Users/giorgiomilitello/Desktop/Extension project/FOMC_Bauer_Swanson.xlsx'
mps_data = pd.read_excel(mps_path, sheet_name='High-Freq FOMC Announcemt Surp')
mps_data['Date'] = pd.to_datetime(mps_data['Date'])

# Determine the last date for which MPS data is available
last_mps_date = mps_data['Date'].max()

# Map MPS values to dates
mps_mapping = mps_data.set_index('Date')['MPS'].to_dict()

# Define horizons and results storage
H = 30  # Maximum horizon
all_results = []

# Dictionary to store data for each tenor
iv_dataframes = {}

# Process each tenor sheet
for tenor in excel_data.sheet_names:
    df = pd.read_excel(file_path, sheet_name=tenor)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Shock'] = df['Date'].isin(fomc_dates['End']).astype(int)
    df['MPS'] = df['Date'].map(mps_mapping).fillna(0)
    iv_dataframes[tenor] = df
    print(f"Processed tenor: {tenor}")

# Create output figures
figures_root = 'Figures'
os.makedirs(figures_root, exist_ok=True)

currency_figures_folder = os.path.join(figures_root, currency)
os.makedirs(currency_figures_folder, exist_ok=True)

# Perform analysis for each tenor
for tenor, df in iv_dataframes.items():
    if df is None or 'Shock' not in df.columns:
        continue

    deltas = [col for col in df.columns if col not in ['Shock', 'MPS', 'Date', 'Option',]]
    event_dates = df.loc[(df['Shock'] == 1) & (df['Date'] <= last_mps_date), 'Date']

    for delta in deltas:
        if delta == 50:  # Skip ATM options
            continue

        cumulative_beta, cumulative_se = [], []

        for h in range(-5, H + 1):
            log_differences = []
            mps_values = []

            for event_date in event_dates:
                # Select only the rows corresponding to the event date
                rr_minus_1 = df.loc[(df['Date'] == event_date + pd.Timedelta(days=-1)) & (df['Option'] == "Call"), moneyness].values - df.loc[(df['Date'] == event_date + pd.Timedelta(days=-1)) & (df['Option'] == "Put"), moneyness].values
                rr_t_h = df.loc[(df['Date'] == event_date + pd.Timedelta(days=h)) & (df['Option'] == 'Call'), delta].values - df.loc[(df['Date'] == event_date + pd.Timedelta(days=h)) & (df['Option'] == 'Put'), delta].values

                # Ensure we have valid values for risk reversal calculation
                if len(call_iv_t_minus_1) > 1 and len(put_iv_t_minus_1) > 0 and len(call_iv_t_h) > 1 and len(put_iv_t_h) > 0:
                    rr_t_minus_1 = call_iv_t_minus_1[1] - put_iv_t_minus_1[0]
                    rr_t_h = call_iv_t_h[1] - put_iv_t_h[0]

                    if pd.notna(rr_t_minus_1) and pd.notna(rr_t_h) and rr_t_minus_1 > 0 and rr_t_h > 0:
                        log_diff = np.log(rr_t_h) - np.log(rr_t_minus_1)
                        log_differences.append(log_diff)
                        mps_values.append(df.loc[df['Date'] == event_date, 'MPS'].values[0])

            if log_differences and mps_values:
                df_reg = pd.DataFrame({'Y_h': log_differences, 'Shock_MPS': mps_values})
                Y = df_reg['Y_h']
                X = sm.add_constant(df_reg['Shock_MPS'])

                try:
                    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                    beta_h = model.params['Shock_MPS']
                    se_h = model.bse['Shock_MPS']
                except:
                    beta_h, se_h = np.nan, np.nan
            else:
                beta_h, se_h = np.nan, np.nan

            cumulative_beta.append(np.nansum(cumulative_beta) + beta_h)
            cumulative_se.append(np.sqrt(np.nansum(np.array(cumulative_se) ** 2) + se_h ** 2))

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(range(-5, H + 1), cumulative_beta, label=f'{delta}-Delta Risk Reversal', marker='o')
        plt.fill_between(range(-5, H + 1), np.array(cumulative_beta) - 1.96 * np.array(cumulative_se),
                         np.array(cumulative_beta) + 1.96 * np.array(cumulative_se), alpha=0.3)
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='red', linestyle='-', label='Event Date (h=0)')
        plt.title(f'Cumulative IRF for {delta}-Delta Risk Reversal (Tenor: {tenor})')
        plt.xlabel('Horizon (h)')
        plt.ylabel('Cumulative Treatment Effect (Log Differences)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(currency_figures_folder, f'{tenor}_{delta}_delta_risk_reversal.png'))
        plt.close()
        print(f"Figure saved for tenor: {tenor}, delta: {delta}")