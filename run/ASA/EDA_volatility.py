import pandas as pd
import yfinance as yf

import NonlinearML.lib.io as io
import NonlinearML.lib.utils as utils
import NonlinearML.plot.plot as plot


# Set input path for beta correlation data
beta_corr_path = '/mnt/mainblob/nonlinearML/data/ASA/csv/20200427_CORR.xls'
beta_corr_date = 'smDate'
beta_corr = ['PM6M', 'PM12M']

# Set output path
output_path = 'output/ASA/EDA/volatility/' 

# Set logging configuration
utils.create_folder(output_path)
io.setConfig(path=output_path, filename="log.txt")


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    io.title('Examining volatility')
    io.message('Output path: %s' %output_path)

    #---------------------------------------------------------------------------
    # VIX index
    #---------------------------------------------------------------------------
    io.title('Examining VIX index')
    # Import vix data from yahoo as dataframe
    VIX_TICKER = "^VIX"
    VIX_PERIOD = "8760d"
    vix = yf.Ticker(VIX_TICKER)
    df = vix.history(period=VIX_PERIOD)

    # Calculate approximation to error
    df['High-Low'] = (df['High'] - df['Low'])/2

    # Plot coefficients as a function of time
    plot.plot_errorbar(
        df=df, x='index', y='Close', error='High-Low',
        x_label="Time", y_label="VIX", figsize=(20,8),
        filename=output_path+"VIX" , legend=False, grid='both',
        fmt='o', markerfacecolor='dodgerblue',
        linewidth=0, markeredgecolor='black',
        markersize=5, elinewidth=1, ecolor='gray')

    #---------------------------------------------------------------------------
    # Beta correlation
    #---------------------------------------------------------------------------
    io.title('Beta correlation')
    df_beta = pd.read_excel(beta_corr_path)

    # Convert datetime64[D] to datetime.date
    df_beta[beta_corr_date] = df_beta[beta_corr_date].astype('O')

    # Plot beta correlation
    plot.plot_line_multiple_cols(
        df=df_beta, x=beta_corr_date,
        list_y=beta_corr, legends=beta_corr, grid='both',
        x_label='Time', y_label='Corr', figsize=(20,6),
        filename=output_path+"beta_corr")


















