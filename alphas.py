import numpy as np
import pandas as pd

# Helper Functions Definitions

def abs_func(x):
    return np.abs(x)

def log_func(x):
    return np.log(x)

def sign_func(x):
    return np.sign(x)

def rank(x):
    return pd.Series(x).rank(method='min').values

def delay(series, d):
    return series.shift(d)

def correlation(series1, series2, d):
    return series1.rolling(window=d).corr(series2)

def covariance(series1, series2, d):
    return series1.rolling(window=d).cov(series2)

def scale(series, a=1):
    return (series / np.sum(np.abs(series))) * a

def delta(series, d):
    return series - delay(series, d)

def signedpower(x, a):
    return np.sign(x) * (np.abs(x) ** a)

def decay_linear(series, d):
    weights = np.arange(1, d + 1)[::-1]  # Linearly decaying weights
    weights = weights / weights.sum()  # Normalize 
    return series.rolling(window=d).apply(lambda x: np.dot(x, weights), raw=False)

def indneutralize(series, group):
    group_mean = series.groupby(group).transform('mean')
    return series - group_mean

def ts_min(series, d):
    return series.rolling(window=d).min()

def ts_max(series, d):
    return series.rolling(window=d).max()

def ts_argmax(series, d):
    return series.rolling(window=d).apply(lambda x: x.idxmax(), raw=False)

def ts_argmin(series, d):
    return series.rolling(window=d).apply(lambda x: x.idxmin(), raw=False)

def ts_rank(series, d):
    return series.rolling(window=d).apply(lambda x: x.rank().iloc[-1], raw=False)

def sum_func(series, d):
    return series.rolling(window=d).sum()

def product(series, d):
    return series.rolling(window=d).apply(lambda x: np.prod(x), raw=True)

def stddev(series, d):
    return series.rolling(window=d).std()

def alpha_1(df):
    close = df['Close']
    returns = close.pct_change()
    
    condition = returns < 0
    
    adjusted_values = np.where(condition, stddev(returns, 20), close)
    
    powered_values = signedpower(adjusted_values, 2)
    
    argmax_values = ts_argmax(pd.Series(powered_values), 5)
    print(argmax_values)
    
    ranked_values = rank(argmax_values)
    
    alpha_values = ranked_values - 0.5
    
    return alpha_values

def momentum(df):
    close = df['Close']
    returns = close.pct_change()

    alpha_values = returns.rolling(20).sum()
    return alpha_values