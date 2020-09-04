import numpy as np
from .core import *


def macd_vi(high,
            low,
            close,
            window_vi=200,
            macd_fast=12 * 4,
            macd_slow=26 * 4,
            signal_period=2):
    vi_plus, vi_minus = vortex_index(high, low, close, window_vi)
    vi_diff = vi_plus - vi_minus
    vi_sum = np.cumsum(vi_diff[window_vi + 1:])
    macd_vi = offset(
        macd(vi_sum, macd_fast=12 * 4, macd_slow=26 * 4, signal_period=2)[2],
        window_vi + 1)
    macd_vi[:window_vi] = np.nan
    return macd_vi
