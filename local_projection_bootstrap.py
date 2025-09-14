import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

# Load the Excel file
file_path = '/Users/giorgiomilitello/Desktop/Extension project/USDJPY.xlsx'
excel_data = pd.ExcelFile(file_path)

# Define moneyness levels
moneyness_levels = [5, 10, 18, 25, 35, 50]  # Ensure 50 is included only once

# Dictionary to store data for each tenor
iv_dataframes = {}

# Iterate over each sheet (tenor) to collect IV data for each tenor
for tenor in excel_data.sheet_names:
    df = pd.read_excel(file_path, sheet_name=tenor)

    # Generate feature names for this tenor
    tenor_features = []
    for moneyness in moneyness_levels:
        if moneyness == 50:
            tenor_features.append(f'{tenor}-50 Delta (ATM)')
        else:
            tenor_features.append(f'{tenor}-{moneyness} Delta Put')
            tenor_features.append(f'{tenor}-{moneyness} Delta Call')

    # Collect IV data for this specific tenor
    all_iv_data = []
    all_dates = []

    for date in df['Date'].unique():
        filtered_data_put = df[(df['Date'] == date) & (df['Option'] == 'Put')]
        filtered_data_call = df[(df['Date'] == date) & (df['Option'] == 'Call')]

        if not filtered_data_put.empty and not filtered_data_call.empty:
            iv_row = []
            for moneyness in moneyness_levels:
                # Retrieve put and call IV values for the current moneyness level
                put_iv = filtered_data_put[moneyness].values if moneyness in filtered_data_put.columns else []
                call_iv = filtered_data_call[moneyness].values if moneyness in filtered_data_call.columns else []

                # Check if moneyness is ATM (50 delta), only use one value
                if moneyness == 50:
                    iv_row.append(put_iv[0] if put_iv.size > 0 else np.nan)
                else:
                    # Append both put and call IVs if available
                    iv_row.append(put_iv[0] if put_iv.size > 0 else np.nan)
                    iv_row.append(call_iv[0] if call_iv.size > 0 else np.nan)

            # Append the row and date if the length matches the features for this tenor
            if len(iv_row) == len(tenor_features):
                all_iv_data.append(iv_row)
                all_dates.append(date)
            else:
                print(f"Skipping: IV row length ({len(iv_row)}) does not match expected feature length ({len(tenor_features)}) for Date: {date}, Tenor: {tenor}")
                continue

    # Create a DataFrame for this tenor
    try:
        iv_dataframes[tenor] = pd.DataFrame(all_iv_data, index=pd.to_datetime(all_dates), columns=tenor_features)
        print(f"IV DataFrame for tenor {tenor} created successfully.")
    except ValueError as e:
        print(f"Error creating IV DataFrame for tenor {tenor}: {e}")
        iv_dataframes[tenor] = None

# Debugging outputs to verify data consistency
for tenor, df in iv_dataframes.items():
    if df is not None:
        print(f"DataFrame for tenor {tenor}:\n{df.head()}")

# Load FOMC dates CSV file
fomc_dates = pd.read_csv('/Users/giorgiomilitello/Desktop/Extension project/FOMCdates.csv')
fomc_dates['End'] = pd.to_datetime(fomc_dates['End'])

# Add FOMC shock indicator to each tenor's DataFrame
for tenor, df in iv_dataframes.items():
    if df is not None:
        df['Shock'] = df.index.isin(fomc_dates['End']).astype(int)
        print(f"Shock indicator added for tenor {tenor}.")

# Define horizons and results storage
H = 30  # Maximum horizon
all_results = []

for tenor, df in iv_dataframes.items():
    if df is None or 'Shock' not in df.columns:
        print(f"Skipping tenor {tenor} due to missing data.")
        continue

    print(f"Processing tenor: {tenor}")
    features = [col for col in df.columns if col != 'Shock']

    # Identify FOMC event dates (where Shock = 1)
    event_dates = df.index[df['Shock'] == 1]

    for feature in features:
        beta_h_values = []
        se_h_values = []

        for h in range(H + 1):
            # Prepare data for this horizon
            differences = []
            for event_date in event_dates:
                # Get IV_t-1 (one day before the event)
                iv_t_minus_1 = df.loc[event_date - pd.Timedelta(days=1), feature] if event_date - pd.Timedelta(days=1) in df.index else np.nan

                # Get IV_t+h (h days after the event)
                iv_t_plus_h = df.loc[event_date + pd.Timedelta(days=h), feature] if event_date + pd.Timedelta(days=h) in df.index else np.nan

                # Calculate the difference
                if not np.isnan(iv_t_minus_1) and not np.isnan(iv_t_plus_h):
                    differences.append(iv_t_plus_h - iv_t_minus_1)

            # Create a DataFrame for regression
            if differences:
                df_reg = pd.DataFrame({
                    'Y_h': differences,
                    'Shock': [1] * len(differences)  # Treatment indicator
                })

                # Perform regression
                Y = df_reg['Y_h']
                X = sm.add_constant(df_reg['Shock'])

                try:
                    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                    beta_h = model.params['Shock']
                    se_h = model.bse['Shock']
                except Exception as e:
                    print(f"Error in regression for {feature}, horizon {h}: {e}")
                    beta_h, se_h = np.nan, np.nan
            else:
                beta_h, se_h = np.nan, np.nan

            beta_h_values.append(beta_h)
            se_h_values.append(se_h)

        # Store results for this feature
        for h, (beta_h, se_h) in enumerate(zip(beta_h_values, se_h_values)):
            all_results.append({
                'Tenor': tenor,
                'Feature': feature,
                'Horizon': h,
                'Beta_h': beta_h,
                'SE_h': se_h
            })

        # Plot IRF
        plt.figure(figsize=(10, 6))
        plt.plot(range(H + 1), beta_h_values, label='Beta_h (Treatment Effect)', marker='o')
        plt.fill_between(
            range(H + 1),
            np.array(beta_h_values) - 1.96 * np.array(se_h_values),
            np.array(beta_h_values) + 1.96 * np.array(se_h_values),
            alpha=0.3, label='95% CI'
        )
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'IRF for {feature} (Tenor: {tenor})')
        plt.xlabel('Horizon (h)')
        plt.ylabel('Treatment Effect')
        plt.legend()
        plt.grid(True)
        plt.show()

# Save results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('/Users/giorgiomilitello/Desktop/Extension project/significant_results.csv', index=False)