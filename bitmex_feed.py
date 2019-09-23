import time
import helpers as hlp
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from pyti import bollinger_bands
import ccxt
import numpy as np
from timeit import default_timer as timer
bitmex = ccxt.bitmex()

count = 20  # count in days in the past
gran = '1h'
symbol = 'BTC/USD'
price_prom = 0.15
params = {
    "count": count,
    "granularity": gran
}
if __name__ == "__main__":
    df = hlp.get_bitmex(symbol, gran, count, m15=False)
    # dfdt = pd.Series(df['close'].values, index=pd.DatetimeIndex(
    #    start='2017-01-1', periods=len(df['close']), freq='1d'))
    # dfseas = seasonal_decompose(dfdt)
    # finding BB%
    dfflist = df['close'].tolist()
    print(df.head())
    bbl = bollinger_bands.percent_b(dfflist, 20, 2.0)
    bbdivs_bear, bbdivs_bull, bpf, pbf = hlp.divergence2(
        bbl, df, prominence=5, hei=20)
    # SRSI and its peaks/bottoms
    K, D = hlp.StochRSI(df, price='close')
    kdivs_bear, kdivs_bull, kbf, kpf = hlp.divergence2(K, df,
                                                       prominence=5, hei=20)

    # peaks of close : resistance and support levels
    dfflist2 = df['close'].values
    prom = price_prom * (np.max(dfflist2) - np.min(dfflist2))
    print(prom)
    lines = hlp.hrlines(df, prominence=prom)

    dfohlc = df
    del dfohlc['DateTime']
    dfohlc = dfohlc.reset_index()
    dfohlc.reindex(range(0, len(df['close'])))

    # plotting
    f, axarr = plt.subplots(2, sharex=True)
    for x in lines.index:
        axarr[0].hlines(y=lines['peak'].iloc[x], xmin=x, xmax=len(
            df['close']) - 1, colors=lines['color'].iloc[x])
    candlestick_ohlc(axarr[0], dfohlc.values,
                     colorup='green', colordown='red')
    # axarr[0].plot(df['close'] * bbdivs_bull['div'].shift(1).values, 'g^')
    # axarr[0].plot(df['close'] * bbdivs_bear['div'].shift(1).values, 'rv')
    axarr[0].plot(df['close'] * kdivs_bear['div'].shift(1).values, 'ro')
    axarr[0].plot(df['close'] * kdivs_bull['div'].shift(1).values, 'go')
    axarr[0].set_title(symbol + ' ' + gran)
    # axarr[1].set_title('Bollinger Bands Percent Bandwidth')
    # axarr[1].plot(bbl)
    # axarr[1].plot(bbl * bpf, 'rx')
    # axarr[1].plot(bbl * pbf, 'gx')
    axarr[1].set_title('SRSI')
    axarr[1].plot(K)
    axarr[1].plot(D)
    axarr[1].plot(kbf * K, 'r+')
    axarr[1].plot(kpf * K, 'g+')
    plt.show()
