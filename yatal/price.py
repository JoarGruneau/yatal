import numpy as np

from .core import rolling_func, offset, divide, replace_nan


def true_price(high, low, close):
    return (high + low + close) / 3


def change(price, window=1):
    return offset(price[window:] / price[:-window], window)


def diff(series, window=1):
    return offset(series[window:] / series[:-window], window)


def aroon(high, low, window=20):
    aroon_up = rolling_func(high, window + 1, [np.argmax])[0] / window
    aroon_down = rolling_func(low, window + 1, [np.argmin])[0] / window
    return aroon_down * 100, aroon_up * 100


def aroon_oscillator(high, low, window=20):
    aroon_down, aroon_up = aroon(high, low, window=window)
    return aroon_up - aroon_down


def b_bands(price, window=20, multiplier=2):
    mean, std = rolling_func(price, window, [np.mean, np.std])
    lower = mean - multiplier * std
    upper = mean + multiplier * std
    return upper, mean, lower


def b_bands_percent(price, window=20, multiplier=2):
    upper, _, lower = b_bands(price, window=window, multiplier=multiplier)
    out = divide(price - lower, upper - lower)
    return replace_nan(out)


def vortex_index(high, low, close, window=20):
    true_range = np.maximum.reduce(
        [high[1:] - low[1:], high[1:] - close[:-1], low[1:] - close[:-1]])
    true_range = offset(rolling_func(true_range, window, [np.sum])[0], 1)
    uptrend = offset(np.abs(high[1:] - low[:-1]), 1)
    downtrend = offset(np.abs(low[1:] - high[:-1]), 1)

    vi_plus = divide(rolling_func(uptrend, window, [np.sum])[0], true_range)
    vi_minus = divide(rolling_func(downtrend, window, [np.sum])[0], true_range)
    vi_plus[:window] = np.nan
    vi_minus[:window] = np.nan
    return replace_nan(vi_plus), replace_nan(vi_minus)
