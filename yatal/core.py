import numpy as np
from numba import jit, float64


def offset(series, window, axis=0, value=np.nan):
    out_shape = list(series.shape)
    out_shape[axis] += window
    out = np.empty(out_shape)
    out[window:] = series
    out[:window] = value
    return out


def rolling_func(series, window, funcs):
    orig_shape = series.shape
    shape = orig_shape[:-1] + (orig_shape[-1] - window + 1, window)
    strides = series.strides + (series.strides[-1], )
    stride_series = np.lib.stride_tricks.as_strided(series,
                                                    shape=shape,
                                                    strides=strides)
    return [offset(func(stride_series, axis=-1)[1:], window) for func in funcs]


def replace_nan(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr[idx]
    return out


def divide(dividend, divisor):
    divisor = divisor.copy()
    divisor[divisor == 0] = np.nan
    return dividend / divisor


@jit((float64[:], float64, float64), nopython=True, nogil=True)
def _ema(series, start, alpha):
    n = series.shape[0]
    ewma = np.empty(n, dtype=float64)
    ewma[0] = start
    for i in range(1, n):
        ewma[i] = series[i] * alpha + ewma[i - 1] * (1 - alpha)
    return ewma
