# access_token = "fa460dd7feedfc3de8b4737c29ae9b37-44c69dafc4645c0fb525215770210b62"
# accountID = "101-004-9745234-003"
import srlevels as hlp
import numpy as np
import matplotlib.pyplot as plt
from pyti import bollinger_bands
from mpl_finance import candlestick_ohlc

# from statsmodels.tsa.seasonal import seasonal_decompose
# import pandas as pd

count = 24 * 20  # count in bars
gran = 'H1'
symbol = 'EUR_USD'
height = 70
price_prom = 0.1
indic_prom = 20

params = {
    "count": count,
    "granularity": gran
}


if __name__ == "__main__":
    df = hlp.OandaDataFrameFactory(symbol, params)
    df = df.reset_index()
    del df['DateTime']
    # df = df.tail(96 * 6)
    # df = df.head(10*96)
    # dfdt = pd.Series(df['close'].values, index=pd.DatetimeIndex(
    #     start='2017-01-1', periods=len(df['close']), freq='1d'))
    # seasonalresult = seasonal_decompose(dfdt)
    # finding BB%
    dfflist = df['close'].tolist()
    bbl = bollinger_bands.percent_b(dfflist, 20, 2.0)
    bbdivs_bear, bbdivs_bull, bpf, pbf = hlp.divergence2(
        bbl, df, prominence=indic_prom, hei=height)

    # SRSI and its peaks/bottoms
    K, D = hlp.StochRSI(df, price='close')

    kdivs_bear, kdivs_bull, kbf, kpf = hlp.divergence2(K, df,
                                                       prominence=20)

    # peaks of close : resistance and support levels
    dfflist2 = df['close'].values
    prom = price_prom * (np.max(dfflist2) - np.min(dfflist2))
    lines = hlp.hrlines(df, prominence=prom)
    dfohlc = df
    dfohlc = dfohlc.reset_index()
    dfohlc.reindex(range(0, len(df['close'])))
    f, axarr = plt.subplots(2, sharex=True)
    for x in lines.index:
        axarr[0].hlines(y=lines['peak'].iloc[x], xmin=x, xmax=len(
            df['close']) - 1, colors=lines['color'].iloc[x])
    candlestick_ohlc(axarr[0], dfohlc.values,
                     colorup='green', colordown='red')
    axarr[0].plot(df['close'] * bbdivs_bull['div'].shift(1).values, 'g^')
    axarr[0].plot(df['close'] * bbdivs_bear['div'].shift(1).values, 'rv')
    print(bbdivs_bull.tail(100))
    axarr[0].plot(df['close'] * kdivs_bear['div'].shift(1).values, 'ro')
    axarr[0].plot(df['close'] * kdivs_bull['div'].shift(1).values, 'go')
    axarr[0].set_title(symbol + ' ' + gran)
    axarr[1].set_title('SRSI')
    # axarr[1].plot(bbl)
    # axarr[1].plot(bbl * bpf, 'gx')
    # axarr[1].plot(bbl * pbf, 'rx')
    axarr[1].plot(K)
    axarr[1].plot(D)
    axarr[1].plot(kbf * K, 'r+')
    axarr[1].plot(kpf * K, 'g+')
    # seasonalresult.plot()

    plt.show()
