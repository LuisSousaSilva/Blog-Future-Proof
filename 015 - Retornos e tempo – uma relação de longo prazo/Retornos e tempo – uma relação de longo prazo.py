# In[1]:
# # importing libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import cufflinks as cf
import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np
import quandl
import plotly
import time
import csv

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Markdown, display
from matplotlib.ticker import FuncFormatter
from pandas.core.base import PandasObject
from datetime import datetime

# Setting pandas dataframe display options
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 800)
pd.set_option('max_colwidth', 800)

# Set plotly offline
init_notebook_mode(connected=True)

# Set matplotlib style
plt.style.use('seaborn')

# Set cufflinks offline
cf.go_offline()

# Defining today's Date
from datetime import date
today = date.today()

## My functions
def compute_drawdowns(dataframe):
    '''
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    '''
    return (dataframe / dataframe.cummax() -1) * 100

dimensions=(990, 500)

def merge_time_series(df_1, df_2):
    df = df_1.merge(df_2, how='left', left_index=True, right_index=True)
    return df

def filter_by_years(dataframe, years=0):
    
    last_date = dataframe.tail(1).index
    year_nr = last_date.year.values[0]
    month_nr = last_date.month.values[0]
    day_nr = last_date.day.values[0]
    
    if month_nr == 2 and day_nr == 29 and years % 4 != 0:
        new_date = str(year_nr - years) + '-' + str(month_nr) + '-' + str(day_nr-1)        
    else:
        new_date = str(year_nr - years) + '-' + str(month_nr) + '-' + str(day_nr)
    
    df = dataframe.loc[new_date:]
    
    dataframe = pd.concat([dataframe.loc[:new_date].tail(1), dataframe.loc[new_date:]])
    # Delete repeated days
    dataframe = dataframe.loc[~dataframe.index.duplicated(keep='first')]

    return dataframe

def compute_rolling_cagr(dataframe, years):
    rolling_result = []
    number = len(dataframe)

    for i in np.arange(1, number + 1):
        df = dataframe.iloc[:i]
        df = filter_by_years(df, years=years)
        result = (((df.iloc[-1] / df.iloc[0]) ** (1/years) - 1))
        rolling_result.append(result[0])

    final_df = pd.DataFrame(data = rolling_result, index = dataframe.index[0:number], columns = ['Ret'])
    final_df = final_df.loc[dataframe.index[0] + pd.DateOffset(years=years):]
    return final_df    


# In[2]:
SP = yf.download("^GSPC", start="1984-01-01")[['Close']]

# In[3]:
Dates = pd.DataFrame(SP.index)

Dates.transpose().to_csv('SP_dates_20200410.txt',index=False, quoting=csv.QUOTE_ALL, header=False)

pd.DataFrame(round(SP, 2).values).transpose().to_csv('SP_20200410.txt',index=False, quoting=csv.QUOTE_ALL, header=False)

# In[4]:
SP.iplot(dimensions=dimensions, color='royalblue', title='S&P 500 desde 1984-01-01')

# In[5]:
rolling_returns_10 = compute_rolling_cagr(SP, years=10)

# In[6]:
rolling_returns_10 = merge_time_series(SP, rolling_returns_10[['Ret']]) * 100

# In[7]:
rolling_returns_10 = rolling_returns_10['Ret']

# In[8]:
Dates = pd.DataFrame(rolling_returns_10.index)

Dates.transpose().to_csv('rolling_returns_10_dates_20200410.txt',index=False, quoting=csv.QUOTE_ALL, header=False)

pd.DataFrame(round(rolling_returns_10, 2).values).transpose().to_csv('rolling_returns_10_20200410.txt',index=False, quoting=csv.QUOTE_ALL, header=False)

# In[9]:
round(rolling_returns_10, 2).iplot(dimensions=dimensions, title='Retornos Rolantes do S&P 500 a 10 anos', yTitle='Percentagem', color='#8b5cac')



