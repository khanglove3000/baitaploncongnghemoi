"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from stockstats import StockDataFrame as Sdf
import plotly.graph_objs as go
import plotly.figure_factory as ff


from .data import create_dataframe
from .layout import html_layout

from datetime import datetime, timedelta


# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, GRU

import os.path
from os import path


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from .predict import Normalizer, prepare_data, PredictTrainingData, PredictTestData, PrepareDataPredict, PredictUnseenData
from .predict import TimeSeriesDataset, LSTModel, PrepareZoomValidation
from .deteted_anomalies import detetech_anomalies
from .content import noidungbieudo
def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Load DataFrame
    df = create_dataframe()
    data = create_dataframe()
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]
    THRESHOLD = 3.4
    time_steps = 30
    test_test, batthuong, test_mae_loss, test_score_df = detetech_anomalies(train, test, THRESHOLD, time_steps)
  
    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[
         dcc.Tabs(id='tabs', children=[ 
          
         # Tab phân tích dư liệu - start
        dcc.Tab(label='TỔNG QUAN', children=[
        html.Div([
             # Biểu đồ giá cổ phiếu - start
                 html.H2('Biểu đồ giá cổ phiếu', style={'textAlign':'center', 'padding-top':5}),
                  dbc.Row(
                    [ dbc.Col(
                        dcc.Dropdown(
                                id="chonbieudo_giacophieu",
                                options=[
                                    {"label": "line", "value": "Line"},
                                    {"label": "candlestick", "value": "Candlestick"},
                                    #{"label": "Simple moving average", "value": "SMA"},
                                    #{"label": "Exponential moving average", "value": "EMA" },
                                    {"label": "MACD", "value": "MACD"},
                                    {"label": "RSI", "value": "RSI"},
                                ],
                                value="Line",
                                style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"},
                            ),
                        ),
                   dbc.Col(
                        dbc.Button(
                                "Plot",
                                id="nut_chon_bieudo",
                                className="mr-1",
                                n_clicks=1,
                                style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "5%"},
                            ),
                        ),
                   ]),
                    dcc.Graph(id='bieudo_giacophieu'),
                    html.Br(),
                    html.Br(),
                    html.Div(id="noidung_bieudo"),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(id='khoiluonggiaodich'),
                    html.H2('Bảng lịch sử giao dịch', style={'textAlign': 'center'}),
                    # Biểu đồ giá cổ phiếu - end
                    create_data_table(df)
           ],
        )], className="container"),        
        # Tab phân tích dư liệu - end
    
        # Tab du bao gia bang thuat toan ARIMA - start
        dcc.Tab(label='DỰ BÁO GIÁ', children=[
        html.Div([
            html.H2("Dự đoán giá cổ phiếu với Deep Neural Networks", style={"textAlign": "center"}),
            dcc.Dropdown(id='my-dropdowntest',
                          options=[{'label': 'Hòa Phát', 'value': 'HPG'}],
                          value='HPG',
                          style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"}),
            dcc.RadioItems(id="radiopred", value="close", labelStyle={'display': 'inline-block', 'padding': 10},
                           options=[{'label': "Giá đóng", 'value': "close"}], 
                           style={'textAlign': "center", }),
            dcc.Graph(id='traintest'), 
            dcc.Graph(id='preds'),
            dcc.Graph(id='zoomvalidation'),
            dcc.Graph(id='next_trading_day'),
           ],
        )], className="container"),        
        # Tab du bao gia bang thuat toan ARIMA - end

        # Tab phat thien bat thuong bang thuat toan LMST - start
         dcc.Tab(label='BẤT THƯỜNG', children=[
                html.Div([
                html.H2("MÔ HÌNH LSTM CHO PHÁT HIỆN BẤT THƯỜNG", style={"textAlign": "center"}),
                dcc.Dropdown(id='khoiluong_batthuong',
                              options=[{'label': 'Hoa Phat', 'value': 'HPG'}],
                              value='HPG', style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "40%"}),    
                dcc.Graph(id='bieudo_khoiluong_batthuong'),

                # html.H2("TRỰC QUAN DỮ LIỆU TEST",style={"textAlign": "center"}),  
                # dcc.Graph(id='plottestdata'),

                html.H2("ĐÁNH GIÁ MODEL",style={"textAlign": "center"}),  
                dcc.Graph(id='danhgiamodel'),
                
                html.H2("NHẬN BIẾT BẤT THƯỜNG",style={"textAlign": "center"}),  
                dcc.Graph(id='batthuong'),
               
                html.H2('BẢNG NHẬN BIẾT BẤT THƯỜNG', style={'textAlign': 'center'}),
            
                bang_nhanbietbatthuong(batthuong),
                html.Br(),
                html.Br(),
                 
                                         
            ])     
        ], className="container"),
        # Tab phat thien bat thuong bang thuat toan LMST - end

        ])
    ], id="dash-container",
    )
    
    
    
    # callback khoi luong giao dịch bat thuong- start--- 
    @dash_app.callback([Output('bieudo_khoiluong_batthuong', 'figure')],
                [Input('khoiluong_batthuong', 'value')])
    def update_graph(khoiluong_batthuong):
    
        trace = []
        data = create_dataframe()
        trace.append(go.Scatter(x=data["date"],y=data["volume"],mode='lines',
            opacity=0.7,name='khối lượng giao dịch'))
        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400'],
                height=600,
                title="Khối lượng giao dịch",
                xaxis={"title":"Thoi gian",
                    'rangeselector': {'buttons': list([
                                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                                {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},
        
                                                {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                
                                                {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                                {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"volume"},     
               
                plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]
    # callback khoi luong giao dịch bat thuong - end--- 
    
    # # callback danh gia model- start--- 
    # @dash_app.callback([Output('plottestdata', 'figure')],
    #             [Input('khoiluong_batthuong', 'value')])
    # def update_graph(test_mae_loss):
    #     hist_data = [test_mae_loss.reshape(-1)]
    #     group_labels = ['train_mae_loss'] # name of the dataset
    #     colors = ['#A56CC1']
    #     fig = go.Figure()
    #     fig.add_trace(ff.create_distplot(hist_data, group_labels, bin_size=.075, colors=colors, show_rug=False, show_curve=True))
    #     fig.update_layout(bargap=0.01)
    #     return [fig]
    # # callback danh gia model- end--- 

    # callback danh gia model- start--- 
    @dash_app.callback([Output('danhgiamodel', 'figure')],
                [Input('khoiluong_batthuong', 'value')])
    def update_graph(selected_dropdown):
        trace = []

        test_test, batthuong, test_mae_loss, test_score_df = detetech_anomalies(train, test, THRESHOLD, time_steps)

        trace.append(go.Scatter(x=test[time_steps:].date, y=test_score_df.loss,
                    mode='lines',
                    name='Test Loss'))
        trace.append(go.Scatter(x=test[time_steps:].date, y=test_score_df.threshold,
                    mode='lines',
                    name='Threshold'))
        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#375CB1', '#FF7400'],
                height=600,
                title="",
                xaxis={"title":"Thoi gian",
                    'rangeselector': {'buttons': list([
                                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                                {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},
        
                                                {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                
                                                {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                                {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Gia dong"},     
                
                plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]
    # callback danh gia model- end--- 


    # callback nhan biet giao dich bat thuong- start--- 
    @dash_app.callback([Output('batthuong', 'figure')],
                [Input('khoiluong_batthuong', 'value')])
    def update_graph(selected_dropdown):
        trace = []
     
        test_test, batthuong, test_mae_loss, test_score_df  = detetech_anomalies(train, test, THRESHOLD, time_steps)

        trace.append(go.Scatter(x=test_test.date, y=test_test.volume,
                    mode='lines',
                    name='khối lượng'))
        trace.append(go.Scatter(x=batthuong.date, y=batthuong.volume,
                    mode='markers',
                    name='Anomaly'))
        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#375CB1', '#FF7400'],
                height=600,
                title="",
                xaxis={"title":"Thoi gian",
                    'rangeselector': {'buttons': list([
                                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                                {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},
        
                                                {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                
                                                {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                                {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Gia dong"},     
                
                plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]
    # callback nhan biet giao dich bat thuong- end--- 


    # callback bảng lịch sử số liệu giao dịch - start---          
    @dash_app.callback(
        # output
        [Output("bieudo_giacophieu", "figure")],
        # input
        [Input("nut_chon_bieudo", "n_clicks")],
        # state
        [State("chonbieudo_giacophieu", "value")],
    )
    def graph_genrator(n_clicks, chart_name):
        dropdown = {"HPG": "Hoa Phat"}
        if n_clicks >= 1:  # Checking for user to click submit button
            
            # loader = DataLoader(MaCoPhieu,str(start_date), end=str(end_date), data_source='vnd')
                # selecting graph type
                # line plot
        # selecting graph type
            # line plot
            stock = Sdf(df)

            if chart_name == "Line":          
                data=[
                    go.Scatter(x=df.date,y=df.close ,mode='lines',
                    opacity=0.7,name= f'Giá đóng cửa', textposition='bottom center')
                ]
                
                title_text = 'Biểu đồ giá đóng - đường thẳng'
                
            # Candelstick
            if chart_name == "Candlestick":
                avg_30 = df.close.rolling(window=30, min_periods=1).mean()
                avg_50 = df.close.rolling(window=50, min_periods=1).mean()
                close_ma_10 = df.close.rolling(10).mean()
                close_ma_20 = df.close.rolling(20).mean()
                close_ma_100 = df.close.rolling(100).mean()
                data=[
                    go.Candlestick(
                        x=list(df.date),
                        open=list(df.open),
                        high=list(df.high),
                        low=list(df.low),
                        close=list(df.close),
                        name="Candlestick",
                    ),
                    go.Scatter(x=list(df.date), y=list(close_ma_10), name='Đường Trung bình động 10 '),
                    go.Scatter(x=list(df.date), y=list(close_ma_20), name='Đường Trung bình động 20 '),
                    go.Scatter(x=list(df.date), y=list(avg_30), name='Đường Trung bình động 30 '),
                    go.Scatter(x=list(df.date), y=list(avg_50), name='Đường Trung bình động 50 '),
                    go.Scatter(x=list(df.date), y=list(close_ma_100), name='Đường Trung bình động 100 '),

                ]
                title_text = 'Biểu Đồ Nến với Simple Moving Average'
            
                
                
            # Simple Moving Average
            if chart_name == "SMA":
                close_ma_10 = df.close.rolling(10).mean()
                close_ma_15 = df.close.rolling(15).mean()
                close_ma_30 = df.close.rolling(30).mean()
                close_ma_100 = df.close.rolling(100).mean()
                data=[
                    go.Scatter(
                        x=list(df.date), y=list(close_ma_10), name="10 ngày"
                    ),
                    go.Scatter(
                        x=list(df.date), y=list(close_ma_15), name="15 ngày"
                    ),
                    go.Scatter(
                        x=list(df.date), y=list(close_ma_30), name="30 ngày"
                    ),
                    go.Scatter(
                        x=list(df.date), y=list(close_ma_100), name="100 ngày"
                    ),
                ]   
                title_text = 'Simple Moving Average'

                
        
            # Exponential moving average
            if chart_name == "EMA":
                close_ema_10 = df.close.ewm(span=10).mean()
                close_ema_15 = df.close.ewm(span=15).mean()
                close_ema_30 = df.close.ewm(span=30).mean()
                close_ema_100 = df.close.ewm(span=100).mean()
                data=[
                    go.Scatter(
                        x=list(df.date), y=list(close_ema_10), name="10 ngày"
                    ),
                    go.Scatter(
                        x=list(df.date), y=list(close_ema_15), name="15 ngày"
                    ),
                    go.Scatter(
                        x=list(df.date), y=list(close_ema_30), name="30 ngày"
                    ),
                    go.Scatter(x=list(df.date), y=list(close_ema_100), name="100 ngày",
                    ),
                ]
                title_text = 'Exponential moving average'


            # Moving average convergence divergence
            if chart_name == "MACD":
                df["MACD"], df["signal"], df["hist"] = (
                    stock["macd"],
                    stock["macds"],
                    stock["macdh"],
                )

                data=[
                    go.Scatter(x=list(df.date), y=list(df.MACD), name="MACD"),
                    go.Scatter(x=list(df.date), y=list(
                        df.signal), name="Signal"),
                    go.Scatter(
                        x=list(df.date),
                        y=list(df["hist"]),
                        line=dict(color="royalblue", width=2, dash="dot"),
                        name="Hitogram",
                    ),
                ]
                title_text = 'Moving Average Convergence Divergence'

                # Relative strength index
            if chart_name == "RSI":
                rsi_6 = stock["rsi_6"]
                rsi_12 = stock["rsi_12"]
                data=[
                    go.Scatter(x=list(df.date), y=list(
                        rsi_6), name="RSI 6 Day"),
                    go.Scatter(x=list(df.date), y=list(
                        rsi_12), name="RSI 12 Day"),
                ]
                title_text = 'Relative strength index'

                
            figure ={'data':data,
                    'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            height=600,
                            title=title_text,
                            xaxis={"title":"Thời gian",
                                'rangeselector': 
                                    {'buttons': 
                                        list([
                                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                                {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},
        
                                                {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                
                                                {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                                {'step': 'all'}
                                            ])},
                                'rangeslider': {'visible': True}, 'type': 'date'},
                            yaxis={"title":"Giá"} ,  
                            
                            plot_bgcolor='rgba(0,0,0,0)')}
        return  [figure]
        # callback bảng lịch sử số liệu giao dịch - end---     
         
                
    @dash_app.callback(
        Output("noidung_bieudo", "children"),
        [Input("nut_chon_bieudo", "n_clicks")],
        # state
        [State("chonbieudo_giacophieu", "value")],
    )
    def graph_genrator(n_clicks, chart_name):
        if n_clicks >= 1:  # Checking for user to click submit button
            noidung_bieudo = noidungbieudo(chart_name)
        return u'Ý nghĩa: "{}"'.format(noidung_bieudo)
         
    # callback khoi luong giao dịch bat thuong- start--- 
    @dash_app.callback([Output('khoiluonggiaodich', 'figure')],
                [Input('khoiluong_batthuong', 'value')])
    def update_graph(khoiluong_batthuong):
    
        trace = []
        data = create_dataframe()
        trace.append(go.Scatter(x=data["date"],y=data["volume"],mode='lines',
            opacity=0.7,name='khối lượng giao dịch'))
        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400'],
                height=600,
                title="Khối lượng giao dịch",
                xaxis={"title":"Thời gian",
                    'rangeselector': {'buttons': list([
                                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                                {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},
        
                                                {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                                
                                                {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                                {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"volume"},     
               
                plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]
    # callback khoi luong giao dịch bat thuong - end--- 
        

    @dash_app.callback([Output('traintest', 'figure')],
                [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
    def update_graph(stock , radioval):
        dropdown = {"HPG": "HOA PHAT"}
        radio = {"close": "Giá đóng" }
        # Data preparation
        data = create_dataframe()
        data_thangtruoc, dataset_test = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]
        trace = []
        trace.append(go.Scatter(x=data['date'],y=data['close'],mode='lines',
            opacity=0.6,name=f'Huấn Luyện (Train)',textposition='bottom center'))
        trace.append(go.Scatter(x=dataset_test['date'],y= dataset_test['close'], mode='lines',
            opacity=0.7,name=f'Kiểm Tra (Validation)',textposition='bottom center'))
       
        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,
                title=f"CHIA THÀNH BỘ DỮ LIỆU HUẤN LUYỆN VÀ KIỂM TRA ",
                xaxis={"title":"Thời gian",
                    'rangeselector': 
                        {'buttons': 
                            list([
                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                    {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},

                                    {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    
                                    {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                    {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},
                yaxis={"title":"Giá đóng"},     
                plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]


    @dash_app.callback([Output('preds', 'figure')],
                [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
    def update_graph(stock , radioval):
        dropdown = {"HPG": "HOA PHAT"}
        radio = {"close": "Giá đóng" }
        data = create_dataframe()
        data_date = data.date
        data_close_price = data.close.values
        num_data_points = len(data.date)
        data_date = list(data_date)
            

        # normalize
        scaler = Normalizer()
        normalized_data_close_price = scaler.fit_transform(data_close_price)
        split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(normalized_data_close_price, num_data_points, scaler, data_date)

        dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
        dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

        model = LSTModel(dataset_train)
       
        predicted_train =  PredictTrainingData(dataset_train, model)
        predicted_val =  PredictTestData(dataset_val, model)
        data_close_price, to_plot_data_y_train_pred, to_plot_data_y_val_pred = PrepareDataPredict(num_data_points, split_index, scaler, predicted_train, predicted_val, data_date, data_close_price)

        trace = []

        trace.append(go.Scatter(x=data_date,y=data_close_price,mode='lines',
            opacity=0.6,name=f'Giá thực tế',textposition='bottom center'))

        trace.append(go.Scatter(x=data_date,y=to_plot_data_y_train_pred,mode='lines',
            opacity=0.6,name=f'Giá dự đoán (train)',textposition='bottom center'))

        trace.append(go.Scatter(x=data_date,y=to_plot_data_y_val_pred,mode='markers',
            opacity=0.6,name=f'Giá dự đoán (validation)',textposition='bottom center'))

        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"So sánh giá dự đoán với giá thực tế",
                xaxis={"title":"Thời gian",
                    'rangeselector': 
                        {'buttons': 
                            list([
                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                    {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},

                                    {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    
                                    {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                    {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Giá đóng"},     
                    plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]

    @dash_app.callback([Output('zoomvalidation', 'figure')],
                [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
    def update_graph(stock , radioval):
        dropdown = {"HPG": "HOA PHAT"}
        radio = {"close": "Giá đóng" }
        data = create_dataframe()
        data_date = data.date
        data_close_price = data.close.values
        num_data_points = len(data.date)
        data_date = list(data_date)
            

        # normalize
        scaler = Normalizer()
        normalized_data_close_price = scaler.fit_transform(data_close_price)
        split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(normalized_data_close_price, num_data_points, scaler, data_date)

        dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
        dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

        model = LSTModel(dataset_train)
        predicted_train =  PredictTrainingData(dataset_train, model)
        predicted_val =  PredictTestData(dataset_val, model)
        to_plot_data_y_val_subset, to_plot_predicted_val, to_plot_data_date = PrepareZoomValidation(split_index, scaler, predicted_val, data_date, data_y_val)
        trace = []

        
        trace.append(go.Scatter(x=to_plot_data_date,y=to_plot_predicted_val,mode='markers',
            opacity=0.6,name=f'Giá dự đoán',textposition='bottom center'))

        trace.append(go.Scatter(x=to_plot_data_date,y=to_plot_data_y_val_subset,mode='lines',
            opacity=0.6,name=f'Giá thực tế',textposition='bottom center'))

        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,
                title=f"Phóng to để kiểm tra giá dự đoán trên tỷ lệ dữ liệu xác thực",
                xaxis={"title":"Thời gian",
                    'rangeselector': 
                        {'buttons': 
                            list([
                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                    {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},

                                    {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    
                                    {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                    {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},
                    yaxis={"title":"Giá đóng"},     
                    plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]

    @dash_app.callback([Output('next_trading_day', 'figure')],
                [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
    def update_graph(stock , radioval):
        dropdown = {"HPG": "HOA PHAT"}
        radio = {"close": "Giá đóng" }
        # Data preparation
        data = create_dataframe()
        data_date = data.date
        data_close_price = data.close.values
        num_data_points = len(data.date)
        data_date = list(data_date)

        # normalize
        scaler = Normalizer()
        normalized_data_close_price = scaler.fit_transform(data_close_price)
        split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(normalized_data_close_price, num_data_points, scaler, data_date)

        dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
        dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

        model = LSTModel(dataset_train)

        predicted_train =  PredictTrainingData(dataset_train, model)
        predicted_val =  PredictTestData(dataset_val, model)
        plot_date_test, to_plot_data_y_val, to_plot_data_y_val_pred, to_plot_data_y_test_pred, prediction= PredictUnseenData(data_x_unseen, model, scaler, data_y_val, predicted_val, data_date )

        trace = []
        trace.append(go.Scatter(x=plot_date_test, y=to_plot_data_y_val, mode='lines+markers',
            opacity=0.7,name=f'Giá thực tế',textposition='top right'))
        trace.append(go.Scatter(x=plot_date_test, y=to_plot_data_y_val_pred, mode='lines+markers',
            opacity=0.7,name=f'Giá dự đoán trong quá khứ',textposition='top right'))
        trace.append(go.Scatter(x=plot_date_test, y=to_plot_data_y_test_pred, mode='markers',
            opacity=0.7,name=f'Giá dự đoán cho ngày tiếp theo',textposition='top right'))
        traces = [trace]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,
                title=f"Dự đoán giá đóng cửa của dữ liệu giao dịch tiếp theo<br><sup>{prediction} VND</sup>",
                xaxis={"title":"Thời gian",
                    'rangeselector': 
                        {'buttons': 
                            list([
                                {'count': 7, 'label': '7 ngày', 'step': 'day', 'stepmode': 'backward'},
                                    {'count': 15, 'label': '15 ngày', 'step': 'day', 'stepmode': 'backward'},

                                    {'count': 1, 'label': '1 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 3, 'label': '3 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    {'count': 6, 'label': '6 tháng', 'step': 'month', 'stepmode': 'backward'},
                                    
                                    {'count': 1, 'label': '1 năm', 'step': 'year', 'stepmode': 'backward'},
                                    {'step': 'all'}
                                                        ])},
                    'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Giá đóng"},     
                plot_bgcolor='rgba(0,0,0,0)')}
        return [figure]        
    return dash_app.server



    

def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    df.date = pd.DatetimeIndex(df.date).strftime("%Y-%m-%d")
    table = dash_table.DataTable(
        id="table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        sort_mode="native",
        page_size=10,
        style_data={
                'width': '100px',
                'maxWidth': '100px',
                'minWidth': '100px',
            },
        # style_cell_conditional=[
        #     {
        #         'if': {'column_id': 'date'},
        #         'width': '10%'
        #     },
        # ],
        style_table={
            'overflowX': 'auto'
            }    
    )

    return table

def bang_nhanbietbatthuong(batthuong):
    """Create Dash datatable from Pandas DataFrame."""
    batthuong.date = pd.DatetimeIndex(batthuong.date).strftime("%Y-%m-%d")
    table = dash_table.DataTable(
            id="table_batthuong",
            columns=[{"name": i, "id": i} for i in batthuong.columns],
            data=batthuong.to_dict("records"),
            sort_action="native",
            sort_mode="native",
            page_size=10,
            style_data={
                    'width': '100px',
                    'maxWidth': '100px',
                    'minWidth': '100px',
                },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'date'},
                    'width': '30%'
                },
            ],
            style_table={
                'overflowX': 'auto'
                }
            )
    return table

def smape_kun(y_true, y_pred):
        return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) +  np.abs(y_true))))
    
