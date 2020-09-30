import numpy as np

from core import rolling_func, offset, _ema, divide, replace_nan
from price import rolling_change, true_price


def sma(series, window=20):
    return rolling_func(series, window, [np.mean])[0]


def ema(series, window=20):
    start = np.mean(series[:window])
    alpha = 2 / (window + 1.0)
    out = _ema(series[window:], start, alpha)
    return offset(out, window)


# diffrent from talib
def cci(high, low, close, window=20):
    typical_price = true_price(high, low, close)
    ma_price, std = rolling_func(typical_price, window, [np.mean, np.std])
    cci = divide(typical_price - ma_price, 0.015 * std)
    return replace_nan(cci)


def macd(price, macd_fast=12, macd_slow=26, signal_period=9):
    ema_fast = ema(price, window=macd_fast)
    ema_slow = ema(price, window=macd_slow)
    macd = ema_fast - ema_slow
    signal_line = offset(ema(macd[macd_slow:], window=signal_period),
                         macd_slow)
    histogram = macd - signal_line
    return macd, signal_line, histogram


def rsi(price, window=20):
    gain = rolling_change(price) - 1
    gain[np.isnan(gain)] = 0
    loss = -gain.copy()
    gain[gain <= 0] = 0
    loss[loss <= 0] = 0
    alpha = 1 / float(window)
    avg_gain = offset(_ema(gain[2:], gain[1], alpha), 2)
    avg_loss = offset(_ema(loss[2:], loss[1], alpha), 2)
    rs = divide(avg_gain, avg_loss)
    rsi = 100 - divide(100, 1 + rs)
    rsi[:window] = np.nan
    return replace_nan(rsi)
