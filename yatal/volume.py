import numpy as np

from core import rolling_func, divide, replace_nan


def obv(price, volume):
    obv = volume.copy()
    obv[1:][price[1:] == price[:-1]] = 0
    negative_change = price[1:] < price[:-1]
    obv[1:][negative_change] = -obv[1:][negative_change]
    return obv.cumsum()


def vwma(price, volume, window=20):
    volume_sum = rolling_func(volume, window, [np.sum])[0]
    price_volume = rolling_func(price * volume, window, [np.sum])[0]
    vwma = divide(price_volume, volume_sum)
    return replace_nan(vwma)


def adl(high, low, close, volume):
    flow = divide(2 * close - high - low, high - low) * volume
    flow[np.isnan(flow)] = 0
    return np.cumsum(flow)
