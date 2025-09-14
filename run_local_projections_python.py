import pandas as pd
import localprojections as lp
import os
# Load and preprocess data
df = pd.read_csv("merged_panel_data_options_combined.csv")
print(df.head())
#Standardize RRs so that they always express how cheaper it is to go long in foreign currency vis-a-vis going short the dollar for a given pair.
# List of columns to multiply by -1
#Settings
rr = 'RR_35'
user = 'Net_Dealer'


rr_columns = ['MPS','RR_5', 'RR_10', 'RR_18', 'RR_25', 'RR_35']

# List of target currencies
target_currencies = ['USDJPY', 'USDCAD', 'USDCHF']

# Apply the transformation
df.loc[df['Currency'].isin(target_currencies), rr_columns] *= -1
###############

df["Date"] = pd.to_datetime(df["Date"])
df["Event"] = (df["MPS"] != 0).astype(int)  # threshold variable
df["Pos_MPS"] = (df["MPS"] > 0).astype(int) #Threshold var is equal to 1 for hawkish announcements
df["Neg_MPS"] = (df["MPS"] < 0).astype(int)
df = df.set_index(['Currency', 'Date'])
print(df.head())
print(df.describe())
print(df.columns)



# Drop rows with missing values in key variables
df = df.dropna(subset=['VIX', 'Spot', 'RR_35', 'Net_Dealer', 'Net_Asset_Mgr', 'Net_Lev_Money'])

# Define model inputs
endog = [rr, 'Net_Dealer', 'Net_Asset_Mgr', 'Net_Lev_Money']  # variables to forecast
exog = ['VIX', 'Spot']  # MPS is the shock variable
response = endog.copy()

# Estimation settings
irf_horizon = 20
opt_lags = 4
opt_cov = 'robust'
opt_ci = 0.95

# irf = lp.PanelLP(data=df,  # input dataframe
#                  Y=endog,  # variables in the model
#                  response=response,  # variables whose IRFs should be estimated
#                  horizon=irf_horizon,  # estimation horizon of IRFs
#                  lags=opt_lags,  # lags in the model
#                  varcov=opt_cov,  # type of standard errors
#                  ci_width=opt_ci  # width of confidence band
#                  )
# # for i in endog:
# #     irfplot = lp.IRFPlot(irf=irf,  # take output from the estimated model
# #                          response=[f'{i}'],  # variables
# #                          shock= ['RR_5'],  # ... to shocks from all variables
# #                          n_columns=2,  # max 2 columns in the figure
# #                          n_rows=2,  # max 2 rows in the figure
# #                          maintitle=f'Panel LP: IRFs of {i} to a shock in RR_5',  # self-defined title of the IRF plot
# #                          show_fig=True,  # display figure (from plotly)
# #                          save_pic=False  # don't save any figures on local drive
# #                          )
#
# irfplot = lp.IRFPlot(irf=irf, # take output from the estimated model
#                      response=endog, # plot only response of invest ...
#                      shock=['MPS'], # ... to shocks from all variables
#                      n_columns=2, # max 2 columns in the figure
#                      n_rows=2, # max 2 rows in the figure
#                      maintitle='Panel LP: IRFs of MPS', # self-defined title of the IRF plot
#                      show_fig=True, # display figure (from plotly)
#                      save_pic=False # don't save any figures on local drive
#                      )
threshold = 'Pos_MPS'
response = endog.copy()  # estimate the responses of all variables to shocks from all variables
irf_horizon = 20  # estimate IRFs up to 8 periods ahead
opt_lags = 1  # include 2 lags in the local projections model
opt_cov = 'kernel'  # HAC standard errors
opt_ci = 0.95  # 95% confidence intervals

irf_on, irf_off = lp.ThresholdPanelLPX(
    data=df,  # input dataframe
    Y=endog,  # endogenous variables in the model
    X=exog,  # exogenous variables in the model
    threshold_var=threshold,  # the threshold dummy variable
    response=response,  # variables whose IRFs should be estimated
    horizon=irf_horizon,  # estimation horizon of IRFs
    lags=opt_lags,  # lags in the model
    varcov=opt_cov,  # type of standard errors
     ci_width=opt_ci  # width of confidence band
     )
irfplot = lp.ThresholdIRFPlot(
    irf_threshold_on=irf_on,  # IRF for when the threshold variable takes value 1
    irf_threshold_off=irf_off,  # IRF for when the threshold variable takes value 0
    response=[user],  # plot only response of invest ...
    shock=endog,  # ... to shocks from all variables
    n_columns=2,  # max 2 columns in the figure
    n_rows=2,  # max 2 rows in the figure
    maintitle='Panel LP: IRFs of MPS',  # self-defined title of the IRF plot
    show_fig=True,  # display figure (from plotly)
    save_pic=False  # don't save any figures on local drive
    )


# Turn on legend + modifications to default plots
out_name = f"{rr}_{user}"
out_dir = "Figures/IRF Plots"
png_path = os.path.join(out_dir, f"{out_name}.png")
irfplot.update_layout(showlegend=True)
irfplot.write_image(png_path, width=3000, height=2000, scale=2)

# Each subplot adds traces in this order:
# 0: grey zero line
# 1–3: ON (crimson) mean, LB, UB
# 4–6: OFF (black) mean, LB, UB
# So per subplot, set names on index 1 and 4
for i in range(0, len(irfplot.data), 7):  # step by subplot
    irfplot.data[i+1].name = "Positive MPS"
    irfplot.data[i+4].name = "Negative MPS"

irfplot.show()