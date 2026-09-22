"""策略模块：双均线交叉策略（SMA Crossover）。

规则：
- 短期均线上穿长期均线（金叉）-> 买入（持仓 = 1）
- 短期均线下穿长期均线（死叉）-> 卖出（持仓 = 0）
"""
import pandas as pd


def sma_crossover_signal(close: pd.Series, short_window: int = 20, long_window: int = 60) -> pd.Series:
    """生成持仓信号序列（0 或 1）。

    信号在收盘后产生，实际持仓从下一交易日开始（在回测中统一 shift(1)）。
    """
    if short_window >= long_window:
        raise ValueError("short_window 必须小于 long_window")
    sma_short = close.rolling(short_window).mean()
    sma_long = close.rolling(long_window).mean()
    # diff > 0 表示短期均线在长期均线之上；astype(int) 得到 0/1 持仓
    signal = (sma_short > sma_long).astype(int)
    return signal
