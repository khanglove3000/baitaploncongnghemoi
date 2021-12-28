"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import vnquant.utils as utils
import pandas as pd
import logging as logging
import re
import requests
import time
import numpy as np
from datetime import datetime
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
URL_VND = 'https://www.vndirect.com.vn/portal/thong-ke-thi-truong-chung-khoan/lich-su-gia.shtml'
URL_CAFE = "http://s.cafef.vn/Lich-su-giao-dich-"
HEADERS = {'content-type': 'application/x-www-form-urlencoded', 'User-Agent': 'Mozilla'}

class DataLoader():
    def __init__(self, symbols, start, end, data_source = 'CAFE', minimal = True, *arg, **karg):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.data_source = data_source
        self.minimal = minimal

    def download(self):
        if str.lower(self.data_source) == 'vnd':
            # raise "Data source VND does not remain support, kindly change data_source='cafe'"
            loader = DataLoaderVND(self.symbols, self.start, self.end)
            stock_data = loader.download()
            # logging.info('Data Symbols: {}, start: {}, end: {}'.format(stock_data, start, end))
            # logging.info('Data VND: {}'.format(stock_data))
        else:
            loader = DataLoaderCAFE(self.symbols, self.start, self.end)
            stock_data = loader.download()
            # logging.info('Data CAFE: {}'.format(stock_data))

        if self.minimal:
            # logging.info(stock_data)
            if str.lower(self.data_source) == 'vnd':
                data = stock_data[['high','low','open','close', 'avg', 'volume']]
                return data
            else:
                data = stock_data[['high', 'low', 'open', 'close', 'adjust', 'volume']]
                return data
        else:
            return stock_data

class DataLoadProto():
    def __init__(self, symbols, start, end, *arg, **karg):
        self.symbols = symbols
        self.start = utils.convert_text_dateformat(start, new_type = '%d/%m/%Y')
        self.end = utils.convert_text_dateformat(end, new_type = '%d/%m/%Y')


class DataLoaderVND(DataLoadProto):
    def __init__(self, symbols, start, end, *arg, **karg):
        self.symbols = symbols
        self.start = start
        self.end = end
        super().__init__(symbols, start, end)

    def download(self):
        stock_datas = []
        if not isinstance(self.symbols, list):
            symbols = [self.symbols]
        else:
            symbols = self.symbols

        for symbol in symbols:
            # stock_datas.append(self.download_one(symbol))
            stock_datas.append(self.download_one_new(symbol))

        data = pd.concat(stock_datas, axis=1)
        return data

    def download_one_new(self, symbol):
        start_date = utils.convert_text_dateformat(self.start, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d')
        end_date = utils.convert_text_dateformat(self.end, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d')
        API_VNDIRECT = 'https://finfo-api.vndirect.com.vn/v4/stock_prices/'
        query = 'code:' + symbol + '~date:gte:' + start_date + '~date:lte:' + end_date
        delta = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        params = {
            "sort": "date",
            "size": delta.days + 1,
            "page": 1,
            "q": query
        }
        res = requests.get(API_VNDIRECT, params=params, headers=HEADERS)
        data = res.json()['data']  
        data = pd.DataFrame(data)
        stock_data = data[['date', 'adClose', 'close', 'pctChange', 'average', 'nmVolume',
                        'nmValue', 'ptVolume', 'ptValue', 'open', 'high', 'low']].copy()
        stock_data.columns = ['date', 'adjust', 'close', 'change_perc', 'avg',
                        'volume_match', 'value_match', 'volume_reconcile', 'value_reconcile',
                        'open', 'high', 'low']

        stock_data = stock_data.set_index('date').apply(pd.to_numeric, errors='coerce')
        stock_data.index = list(map(utils.convert_date, stock_data.index))
        stock_data.index.name = 'date'
        stock_data = stock_data.sort_index()
        stock_data.fillna(0, inplace=True)
        stock_data['volume'] = stock_data.volume_match + stock_data.volume_reconcile

        # Create multiple columns
        iterables = [stock_data.columns.tolist(), [symbol]]
        mulindex = pd.MultiIndex.from_product(iterables, names=['Attributes', 'Symbols'])
        stock_data.columns = mulindex

        logging.info('data {} from {} to {} have already cloned!' \
                     .format(symbol,
                             utils.convert_text_dateformat(self.start, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d'),
                             utils.convert_text_dateformat(self.end, origin_type='%d/%m/%Y', new_type='%Y-%m-%d')))

        return stock_data

    def download_one(self, symbol):
        stock_data = pd.DataFrame(columns=['date', 'change_perc1', 'change_perc2',
                                           'open', 'high', 'low', 'close',
                                           'avg', 'volume_match', 'volume_reconcile'])
        last_page = self.get_last_page(symbol)
        # logging.info('Last page {}'.format(last_page))
        for i in range(last_page):
            stock_slice_batch = self.download_batch(i+1, symbol)
            stock_data = pd.concat([stock_data, stock_slice_batch], axis=0)
        stock_data = stock_data.set_index('date').apply(pd.to_numeric, errors='coerce')
        stock_data.index = list(map(utils.convert_date, stock_data.index))
        stock_data.index.name = 'date'
        stock_data = stock_data.sort_index()
        stock_data.fillna(0, inplace=True)
        stock_data['volume'] = stock_data.volume_match + stock_data.volume_reconcile

        # Create multiple columns
        iterables = [stock_data.columns.tolist(), [symbol]]
        mulindex = pd.MultiIndex.from_product(iterables, names=['Attributes', 'Symbols'])
        stock_data.columns = mulindex

        logging.info('data {} from {} to {} have already cloned!' \
                     .format(symbol,
                             utils.convert_text_dateformat(self.start, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d'),
                             utils.convert_text_dateformat(self.end, origin_type='%d/%m/%Y', new_type='%Y-%m-%d')))

        return stock_data

    def download_batch(self, id_batch, symbol):
        form_data = {"model.downloadType": "",
                     "pagingInfo.indexPage": str(id_batch),
                     "searchMarketStatisticsView.symbol": symbol,
                     "strFromDate": self.start,
                     "strToDate": self.end}
        r = requests.post(URL_VND, form_data, headers=HEADERS, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        data_node = soup.find(class_='list_tktt lichsugia')

        dates = []
        change_percents1 = []
        change_percents2 = []
        opens = []
        highs = []
        lows = []
        closes = []
        avgs = []
        adjusts = []
        volume_matchs = []
        volume_reconciles = []
        # logging.info(data_node)
        for i, value in enumerate(data_node.select('div')):
            if i < 10: continue
            value = utils.clean_text(value.text)
            if i % 10 == 0:
                dates.append(value)
            elif i % 10 == 1:
                values = value.split()
                change_percents1.append(values[0])
                change_percents2.append(values[1])
            elif i % 10 == 2:
                opens.append(value)
            elif i % 10 == 3:
                highs.append(value)
            elif i % 10 == 4:
                lows.append(value)
            elif i % 10 == 5:
                closes.append(value)
            elif i % 10 == 6:
                avgs.append(value)
            elif i % 10 == 7:
                adjusts.append(value)
            elif i % 10 == 8:
                volume_matchs.append(value)
            elif i % 10 == 9:
                volume_reconciles.append(value)

        stock_slice_batch = pd.DataFrame(
            {'date': dates, 'change_perc1': change_percents1, 'change_perc2': change_percents2,
             'open': opens, 'high': highs, 'low': lows, 'close': closes,
             'avg': avgs, 'volume_match': volume_matchs, 'volume_reconcile': volume_reconciles})

        return stock_slice_batch

    def get_last_page(self, symbol):
        form_data = {"searchMarketStatisticsView.symbol":symbol,
                    "strFromDate":self.start,
                    "strToDate":self.end}

        r = requests.post(URL_VND, form_data, headers=HEADERS, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        # last_page = utils.extract_number(str(soup.find_all('div', {'class': 'paging'})[-1].select('a')[-1].attrs))
        text_div = soup.find_all('div', {'class': 'paging'})[-1].get_text()
        try:
            last_page = int(text_div.split()[1].split('/')[1])
        except:
            last_page = int(text_div)
        return last_page

class DataLoaderCAFE(DataLoadProto):
    def __init__(self, symbols, start, end, *arg, **karg):
        self.symbols = symbols
        self.start = start
        self.end = end
        super(DataLoaderCAFE, self).__init__(symbols, start, end)

    def download(self):
        stock_datas = []
        if not isinstance(self.symbols, list):
            symbols = [self.symbols]
        else:
            symbols = self.symbols

        for symbol in symbols:
            stock_datas.append(self.download_one(symbol))

        data = pd.concat(stock_datas, axis=1)
        return data

    def download_one(self, symbol):
        stock_data = pd.DataFrame(columns=['date', 'change_perc1', 'change_perc2',
                                           'open', 'high', 'low', 'close',
                                           'avg', 'volume_match', 'volume_reconcile'])

        for i in range(1000):
            stock_slice_batch = self.download_batch(i + 1, symbol)
            stock_data = pd.concat([stock_data, stock_slice_batch], axis=0)
            try:
                date_end_batch = stock_slice_batch.date.values[-1]
            except:
                # start date is holiday or weekend
                break
            is_touch_end = utils.convert_date(self.start, '%d/%m/%Y') == utils.convert_date(date_end_batch, '%d/%m/%Y')
            # logging.info('batch: {}; start date out range: {}; date_end_batch: {}'.format(i + 1, is_touch_end, date_end_batch))
            if is_touch_end:
                break

        stock_data['change_perc1'], stock_data['change_perc2'] = stock_data['change_perc'].apply(utils.split_change_col).str
        if 'change_perc' in stock_data.columns:
            stock_data.pop('change_perc')
        if 'avg' in stock_data.columns:
            stock_data.pop('avg')
            stock_data = stock_data.set_index('date').apply(pd.to_numeric, errors='coerce')
            stock_data.index = list(map(lambda text: utils.convert_date(text, date_type='%d/%m/%Y'), stock_data.index))
            stock_data.index.name = 'date'
            stock_data = stock_data.sort_index()
            stock_data.fillna(0, inplace=True)
            stock_data['volume'] = stock_data.volume_match + stock_data.volume_reconcile


        # Create multiple columns
        iterables = [stock_data.columns.tolist(), [symbol]]
        mulindex = pd.MultiIndex.from_product(iterables, names=['Attributes', 'Symbols'])
        stock_data.columns = mulindex


        logging.info('data {} from {} to {} have already cloned!' \
                     .format(symbol,
                             utils.convert_text_dateformat(self.start, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d'),
                             utils.convert_text_dateformat(self.end, origin_type='%d/%m/%Y', new_type='%Y-%m-%d')))

        return stock_data

    def download_batch(self, id_batch, symbol):
        form_data = {'ctl00$ContentPlaceHolder1$scriptmanager':'ctl00$ContentPlaceHolder1$ctl03$panelAjax|ctl00$ContentPlaceHolder1$ctl03$pager2',
                       'ctl00$ContentPlaceHolder1$ctl03$txtKeyword':symbol,
                       'ctl00$ContentPlaceHolder1$ctl03$dpkTradeDate1$txtDatePicker':self.start,
                       'ctl00$ContentPlaceHolder1$ctl03$dpkTradeDate2$txtDatePicker':self.end,
                       '__EVENTTARGET':'ctl00$ContentPlaceHolder1$ctl03$pager2',
                       '__EVENTARGUMENT':id_batch,
                       '__ASYNCPOST':'true'}
        url = URL_CAFE+symbol+"-1.chn"
        r = requests.post(url, data = form_data, headers = HEADERS, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        # print(soup)
        table = soup.find('table')
        stock_slice_batch = pd.read_html(str(table))[0].iloc[2:, :12]

        stock_slice_batch.columns = ['date', 'adjust', 'close', 'change_perc', 'avg',
                        'volume_match', 'value_match', 'volume_reconcile', 'value_reconcile',
                        'open', 'high', 'low']

        return stock_slice_batch

# loader1 = DataLoaderVND(symbols="VND", start="2021-01-01", end="2021-02-15")
# loader2 = DataLoaderCAFE(symbols="VND", start="2017-01-10", end="2019-02-15")
# loader3 = DataLoader(symbols='VND', start="2018-01-10", end="2018-02-15", minimal=False, data_source='vnd')
# loader3 = DataLoader(symbols='VND', start="2018-01-10", end="2018-02-15", minimal=True, data_source='vnd')
# loader3 = DataLoader(symbols=['VND', 'VCB'], start="2018-01-10", end="2018-02-15", minimal=True, data_source='vnd')
# loader4 = DataLoader(symbols='VND', start="2018-01-10", end="2018-02-15", minimal=False, data_source='cafe')
# loader4 = DataLoader(symbols='VND', start="2018-01-10", end="2018-02-15", minimal=True, data_source='cafe')
# loader4 = DataLoader(symbols=['VND', 'VCB'], start="2018-01-10", end="2018-02-15", minimal=True, data_source='cafe')


class FinanceLoader():
    def __init__(self, symbol, start, end, *arg, **karg):
        self.symbol = symbol
        self.start = start
        self.end = end

    def get_finan_report(self):
        start_time = time.time()
        page = requests.get("https://finfo-api.vndirect.com.vn/v3/stocks/financialStatement?secCodes={}&reportTypes=QUARTER&modelTypes=1,89,101,411&fromDate={}&toDate={}".format(self.symbol, self.start, self.end))
        data = page.json()
        end_time = time.time()
        data_dates = {}
        #print('request time: ', end_time-start_time)
        start_time = time.time()
        for item in data['data']['hits']:
            item = item['_source']
            date = item['fiscalDate']
            itemName = item['itemName']
            itemCode = item['itemCode']
            numericValue = item['numericValue']
            if date not in data_dates:
                data_dates[date] = [[], []]
            else:
                if itemName not in data_dates[date][0]:
                    data_dates[date][0].append(itemName)
                    data_dates[date][1].append(numericValue)
        end_time = time.time()
        #print('access data time: ', end_time-start_time)
        start_time = time.time()
        for i, (date, item) in enumerate(data_dates.items()):
            df_date = pd.DataFrame(data={'index':item[0], date[:7]:item[1]})
            if i == 0:
                df = df_date
            else:
                df = pd.merge(df, df_date, how='inner')
        df.set_index('index', inplace=True)
        end_time = time.time()
        #print('merge time: ', end_time-start_time)
        return df

    def get_business_report(self):
        start_time = time.time()
        page = requests.get("https://finfo-api.vndirect.com.vn/v3/stocks/financialStatement?secCodes={}&reportTypes=QUARTER&modelTypes=2,90,102,412&fromDate={}&toDate={}".format(self.symbol, self.start, self.end))
        data = page.json()
        end_time = time.time()
        data_dates = {}
        #print('request time: ', end_time-start_time)
        start_time = time.time()
        for item in data['data']['hits']:
            item = item['_source']
            date = item['fiscalDate']
            itemName = item['itemName']
            itemCode = item['itemCode']
            numericValue = item['numericValue']
            if date not in data_dates:
                data_dates[date] = [[], []]
            else:
                if itemName not in data_dates[date][0]:
                    data_dates[date][0].append(itemName)
                    data_dates[date][1].append(numericValue)
        end_time = time.time()
        #print('access data time: ', end_time-start_time)
        start_time = time.time()
        for i, (date, item) in enumerate(data_dates.items()):
            df_date = pd.DataFrame(data={'index':item[0], date[:7]:item[1]})
            if i == 0:
                df = df_date
            else:
                df = pd.merge(df, df_date, how='inner')
        df.set_index('index', inplace=True)
        end_time = time.time()
        #print('merge time: ', end_time-start_time)
        return df

    def get_cashflow_report(self):
        start_time = time.time()
        page = requests.get("https://finfo-api.vndirect.com.vn/v3/stocks/financialStatement?secCodes={}&reportTypes=QUARTER&modelTypes=3,91,103,413&fromDate={}&toDate={}".format(self.symbol, self.start, self.end))
        data = page.json()
        end_time = time.time()
        data_dates = {}
        #print('request time: ', end_time-start_time)
        start_time = time.time()
        for item in data['data']['hits']:
            item = item['_source']
            date = item['fiscalDate']
            itemName = item['itemName']
            itemCode = item['itemCode']
            numericValue = item['numericValue']
            if date not in data_dates:
                data_dates[date] = [[], []]
            else:
                if itemName not in data_dates[date][0]:
                    data_dates[date][0].append(itemName)
                    data_dates[date][1].append(numericValue)
        end_time = time.time()
        #print('access data time: ', end_time-start_time)
        start_time = time.time()
        for i, (date, item) in enumerate(data_dates.items()):
            df_date = pd.DataFrame(data={'index':item[0], date[:7]:item[1]})
            if i == 0:
                df = df_date
            else:
                df = pd.merge(df, df_date, how='inner')
        df.set_index('index', inplace=True)
        end_time = time.time()
        #print('merge time: ', end_time-start_time)
        return df

    def get_basic_index(self):
        start_year = int(self.start[:4])
        end_year = int(self.end[:4])
        years = np.arange(start_year, end_year+1, 1)[::-1]
        years = ['{}-12-31'.format(year) for year in years]
        data_dates = {}
        user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36'
        headers = {'User-Agent': user_agent}
        for year in years:
            page = requests.get("https://finfo-api.vndirect.com.vn/v4/ratios?q=code:{}~itemCode:53030,52005,51050,53021,52001,52002,54018,712010,712020,712030,712040~reportDate:{}".format(self.symbol, year), headers=headers)
            data = page.json()
            
            for item in data['data']:
                date = item['reportDate']
                itemName = item['itemName']
                itemCode = item['itemCode']
                value = item['value']
                if date not in data_dates:
                    data_dates[date] = [[], []]
                else:
                    if itemName not in data_dates[date][0] and itemName != "":
                        data_dates[date][0].append(itemName)
                        data_dates[date][1].append(value)

        for i, (date, item) in enumerate(data_dates.items()):
            df_date = pd.DataFrame(data={'index':item[0], date[:7]:item[1]})
            if i == 0:
                df = df_date
            else:
                df = pd.merge(df, df_date, how='inner')

        df.set_index('index', inplace=True)
        return df

def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    #df = pd.read_csv("data/HPG.csv", parse_dates=["date"])
   # input
    MaCoPhieu = 'HPG' 
    end_date = str(datetime.now().date())
    start_date = str(datetime.now().date() - timedelta(days=5* 365))

    # loader = DataLoader(MaCoPhieu,str(start_date), end=str(end_date), data_source='vnd')
  
    loader = DataLoader(MaCoPhieu, start_date, end_date)
    data = loader.download()
    data.columns =  data.columns.droplevel(1)
    data = data.reset_index(level=0, drop=False)
    data = data.set_index('date')
    data['date'] =data.index
    print(data)
    return data

def create_get_business_report():
    MaCoPhieu = 'HPG' 
    end_date = str(datetime.now().date())
    start_date = str(datetime.now().date() - timedelta(days=5* 365))
    loader = FinanceLoader(MaCoPhieu, start_date, end_date, data_source='VND', minimal=True)
    data_business = loader.get_business_report()
    return data_business


def create_get_finan_report():
    MaCoPhieu = 'HPG' 
    end_date = str(datetime.now().date())
    start_date = str(datetime.now().date() - timedelta(days=5* 365))
    loader = FinanceLoader(MaCoPhieu, start_date, end_date, data_source='VND', minimal=True)
    data_finan = loader.get_finan_report()
    return data_finan

def create_get_cashflow_report():
    MaCoPhieu = 'HPG' 
    end_date = str(datetime.now().date())
    start_date = str(datetime.now().date() - timedelta(days=5* 365))
    loader = FinanceLoader(MaCoPhieu, start_date, end_date, data_source='VND', minimal=True)
    data_cash = loader.get_cashflow_report()
    return data_cash

def create_get_basic_index():
    MaCoPhieu = 'HPG' 
    end_date = str(datetime.now().date())
    start_date = str(datetime.now().date() - timedelta(days=5* 365))
    loader = FinanceLoader(MaCoPhieu, start_date, end_date, data_source='VND', minimal=True)
    data_basic = loader.get_basic_index()
    return data_basic