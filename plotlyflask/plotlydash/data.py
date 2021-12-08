"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import vnquant.DataLoader as web

def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv("data/HPG.csv", parse_dates=["date"])
#    # input
#     MaCoPhieu = 'HPG' 
#     end_date = str(datetime.now().date())
#     start_date = str(datetime.now().date() - timedelta(days=5* 365))

#     # loader = DataLoader(MaCoPhieu,str(start_date), end=str(end_date), data_source='vnd')
  
#     loader = web.DataLoader('HPG', start_date, end_date)
#     data = loader.download()
#     data.columns =  data.columns.droplevel(1)
#     data = data.reset_index(level=0, drop=False)
#     data = data.set_index('date')
#     data['date'] =data.index
#     print(data)
    return df
