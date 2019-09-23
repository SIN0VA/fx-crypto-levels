import pandas as pd
import talib as tl
import quandl
import pickle
import datetime
import numpy as np
from oandapyV20 import API
import oandapyV20.endpoints.instruments as v20instruments
from collections import OrderedDict
import ccxt
from scipy.signal import find_peaks


# it won't work replace your own here
oandatoken = "fa460dd7fe3dfc3de824737c29ae9b27-44c69dafc4645c0f5525213770210b62"
oandaaccountID = "101-004-9745234-003"  # here too


def StochSMA(dfLow, dfHigh, dfClose, period=14, fast=3, slow=3):
    K = pd.Series(((dfClose - dfLow.rolling(period).min()) /
                   (dfHigh.rolling(period).max() - dfLow.rolling(period).min())) * 100, name='K')
    K = K.rolling(window=slow).mean()
    D = pd.Series(K.rolling(window=fast).mean(), name='D')
    return (K, D)


def get_quandl_data(quandl_id, start_date="2018-01-06", end_date="2018-12-11"):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}-{}-{}.pkl'.format(
        quandl_id, start_date, end_date).replace('/', '-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} {} {} from cache'.format(
            quandl_id, start_date, end_date))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        quandl.ApiConfig.api_key = "g5CKrkMZpm11emosrJex"
        df = quandl.get(quandl_id, returns="pandas",
                        start_date=start_date, end_date=end_date)
        df.fillna(method='pad', inplace=True)
        #df.replace(0, np.nan, inplace=True)
        df.to_pickle(cache_path)
        df = df.iloc[::-1]
        df = df.dropna()
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


def StochRSI(df, price='close', period=14, fast=3, slow=3):
    rsi = tl.RSI(df[price], 14)
    K, D = StochSMA(rsi, rsi, rsi)
    # upsamK = K.resample('1T')
    # upsamD = D.resample('1T')
    # upsamK = K.interpolate(method='linear')
    # upsamD = D.interpolate(method='linear')
    # KSMA = tl.SMA(upsamK, 3)
    # DSMA = tl.SMA(upsamD, 3)
    return (K, D)


def get_fxcm(symbol, periodicity='H1', start_date=datetime.date(2018, 12, 1), end_date=datetime.date(2018, 12, 14)):
    url = 'https://candledata.fxcorporate.com/'  # This is the base url
    url_suffix = '.csv.gz'  # Extension of the file name
    cache_path = 'FXCM-{}-{}-{}-{}.pkl'.format(symbol,
                                               start_date, end_date,
                                               periodicity).replace('/', '-')
    # find the week of the year for the start
    start_wk = start_date.isocalendar()[1]
    end_wk = end_date.isocalendar()[1]  # find the week of the year for the end
    year = str(start_date.isocalendar()[0])  # pull out the year of the start
    try:
        f = open(cache_path, 'rb')
        data = pickle.load(f)
        print('Loaded FXCM-{}-{}-{}-{} from cache'.format(
            symbol, start_date, end_date, periodicity))
    except (OSError, IOError) as e:
        print('Downloading {} from FXCM'.format(symbol))
        data = pd.DataFrame()
        for i in range(start_wk, end_wk):
            url_data = url + periodicity + '/' + symbol + \
                '/' + year + '/' + str(i) + url_suffix
            print(url_data)
            tempdata = pd.read_csv(url_data, compression='gzip')
            data = pd.concat([data, tempdata])
        data.to_pickle(cache_path)
        data.fillna(method='pad', inplace=True)
        data = data.iloc[::-1]
        data = data.dropna()
        print('Cached {} at {}'.format(symbol, cache_path))
    #data = data.set_index(pd.bdate_range(start_date, periods=120, freq='H'))
    return data


def divergence_bb(df, peaks, price):
    df['div'] = pd.Series(data=np.nan)
    for i in range(1, len(peaks)):
        df['div'].iloc[peaks[i]] = True if (df['bb'].iloc[peaks[i]] > df['bb'].iloc[
            peaks[i - 1]] and df[price].iloc[peaks[i]] < df[price].iloc[peaks[i - 1]]) or (df['bb'].iloc[peaks[i]] < df['bb'].iloc[
                peaks[i - 1]] and df[price].iloc[peaks[i]] > df[price].iloc[peaks[i - 1]]) else np.nan
    return df


def divergence(df, peaks, column='bb', price='close'):
    df['div'] = pd.Series(data=np.nan)
    for i in range(1, len(peaks)):
        df['div'].iloc[peaks[i]] = True if (df[column].iloc[peaks[i]] > df[column].iloc[
            peaks[i - 1]] and df[price].iloc[peaks[i]] < df[price].iloc[peaks[i - 1]]) or (df[column].iloc[peaks[i]] < df[column].iloc[
                peaks[i - 1]] and df[price].iloc[peaks[i]] > df[price].iloc[peaks[i - 1]]) else np.nan
    return df


def divergence2(K, df, pricebear='high', pricebull='low', prominence=10, hei=20):
    ks = pd.Series(K, index=df.index)
    kpeaks, _ = find_peaks(K, height=100 - hei, prominence=prominence)
    # print(kpeaks)
    kp = pd.Series(True, index=kpeaks)
    kbbp = kp.reindex(range(len(df[pricebear])))
    kp = pd.Series(kbbp.values * K, index=df.index)
    kpf = pd.DataFrame(data={'values': ks.tolist(),
                             pricebear: df[pricebear].tolist()})
    kdivs_bear = divergence(kpf, kpeaks, 'values', pricebear)
    kbottoms, _ = find_peaks(-K, height=-hei, prominence=prominence)
    kb = pd.Series(True, index=kbottoms)
    kbb = kb.reindex(range(len(df[pricebull])))
    kb = pd.Series(kbb.values * K, index=df.index)
    kbf = pd.DataFrame(data={'values': ks.tolist(),
                             pricebull: df[pricebull].tolist()})
    kdivs_bull = divergence(kbf, kbottoms, 'values', pricebull)
    return (kdivs_bear, kdivs_bull, kbbp, kbb)


def convrec(r, m):
    """convrec - convert OANDA candle record.

    return array of values, dynamically constructed, corresponding with config in mapping m.
    """
    v = []
    for keys in [x.split(":") for x in m.keys()]:
        _v = r.get(keys[0])
        for k in keys[1:]:
            _v = _v.get(k)
        v.append(_v)

    return v


def OandaDataFrameFactory(symbol, params, colmap=None, conv=None):
    cache_path = 'OANDA-{}-{}-{}.pkl'.format(symbol,
                                             params['granularity'],
                                             params['count']).replace('/', '-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded OANDA-{}-{}-{} from cache'.format(
            symbol, params['granularity'], params['count']))
        # del df['DateTime']
        df = df.apply(pd.to_numeric)
    except (OSError, IOError) as e:
        api = API(access_token=oandatoken)
        r = v20instruments.InstrumentsCandles(instrument=symbol,
                                              params=params)
        api.request(r)
        record_converter = convrec if conv is None else conv
        column_map_ohlcv = OrderedDict([
            ('time', 'DateTime'),
            ('mid:o', 'open'),
            ('mid:h', 'high'),
            ('mid:l', 'low'),
            ('mid:c', 'close'),
        ])
        cmap = column_map_ohlcv if colmap is None else colmap
        df = pd.DataFrame([list(record_converter(rec, cmap))
                           for rec in r.response.get('candles')])
        df.columns = list(cmap.values())
        # df.rename(columns=colmap, inplace=True)  # no need to use rename, cmap
        # values are ordered
        df.set_index(pd.DatetimeIndex(df['DateTime']), inplace=True)
        del df['DateTime']
        # OANDA returns string values: make all numeric

        df = df.apply(pd.to_numeric)
        df.fillna(method='pad', inplace=True)
        df = df.dropna()
        print('Cached {} at {}'.format(symbol, cache_path))
        df.to_pickle(cache_path)
    return df


def get_bitmex(symbol, gran, count, m15=False):
    # count in days
    limitt = count * 24 if gran == '1h' else count * \
        288 if gran == '5m' else 720 if gran == '1m' else count
    enddate = datetime.datetime.now()
    startdate1 = enddate - datetime.timedelta(days=count)
    print(startdate1)
    startdate = startdate1.strftime("%Y-%m-%d %H:%M:%S")
    cache_path = 'BITMEX-{}-{}-{}-{}.pkl'.format(symbol,
                                                 'M15' if m15 == True else gran,
                                                 limitt,
                                                 startdate.split(' ')[0]).replace('/', '-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded BITMEX-{}-{}-{}-{} from cache'.format(
            symbol, gran, limitt, startdate.replace(' ', '-')))
    except (OSError, IOError) as e:
        ohlcv5 = []
        startdate2 = startdate1
        freqq = 576 if gran == '5m' else 720 if (
            gran == '1h' or gran == '1m') else 1
        lim = limitt // freqq
        dayz = 30 if gran == '1h' else 2
        if limitt > freqq and gran != '1d':
            for x in range(0, lim):
                bitmex = ccxt.bitmex()
                print(startdate2)
                ohlcvt = bitmex.fetch_ohlcv(
                    symbol, gran, bitmex.
                    parse8601(startdate2.strftime("%Y-%m-%d %H:%M:%S")),
                    limit=freqq)
                ohlcv5 = ohlcv5 + ohlcvt
                startdate2 = startdate2 + datetime.timedelta(days=dayz)
        else:
            bitmex = ccxt.bitmex()
            ohlcv5 = bitmex.fetch_ohlcv(
                symbol, gran,
                bitmex.parse8601(startdate2.strftime("%Y-%m-%d %H:%M:%S")),
                limit=limitt)

        ohlcv15 = []
        if m15 == True:
            timestamp = 0
            openn = 1
            high = 2
            low = 3
            close = 4
            volume = 5
            for i in range(0, len(ohlcv5) - 2, 3):
                highs = [ohlcv5[i + j][high]
                         for j in range(0, 3) if ohlcv5[i + j][high]]
                lows = [ohlcv5[i + j][low]
                        for j in range(0, 3) if ohlcv5[i + j][low]]
                volumes = [ohlcv5[i + j][volume]
                           for j in range(0, 3) if ohlcv5[i + j][volume]]
                candle = [
                    ohlcv5[i + 0][timestamp],
                    ohlcv5[i + 0][openn],
                    max(highs) if len(highs) else None,
                    min(lows) if len(lows) else None,
                    ohlcv5[i + 2][close],
                    sum(volumes) if len(volumes) else None,
                ]
                ohlcv15.append(candle)
        header = ['DateTime', 'open', 'high', 'low', 'close', 'Volume']
        df = pd.DataFrame(ohlcv15 if m15 == True else ohlcv5, columns=header)
        df['DateTime'] = pd.to_datetime(df['DateTime'], unit='ms')
        print('Cached {} at {}'.format(symbol, cache_path))
        df.to_pickle(cache_path)
    return df


def hrlines(df, price='close', prominence=100):
    close = df[price].values
    closepeaks, _ = find_peaks(close, prominence=prominence)
    closebottoms, _ = find_peaks(-close, prominence=prominence)
    sorte = np.concatenate((closepeaks, closebottoms))
    sorte.sort()
    peakssp = pd.Series(True, index=sorte)
    peakspp = peakssp.reindex(range(len(df[price])))
    peaksp = pd.Series(peakspp.values *
                       df[price].values, index=df.index)
    lines = pd.DataFrame({'peak': peaksp, 'color':
                          ['r' if x > df[price].iloc[-1]
                           else 'g' for x in peaksp.values]}, index=df.index)
    return lines
