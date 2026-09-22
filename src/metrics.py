"""绩效指标模块：计算常用回测评价指标。"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负数，如 -0.35 表示 -35%）。"""
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def performance_summary(returns: pd.Series, equity: pd.Series) -> dict:
    """输入日收益率序列与净值曲线，返回常用绩效指标。"""
    returns = returns.dropna()
    n = len(returns)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    # 年化收益：按交易日复利折算
    ann_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1) if n > 0 else np.nan
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # 夏普比率（假设无风险利率为 0，简化处理）
    sharpe = float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.std() > 0 else np.nan

    mdd = max_drawdown(equity)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else np.nan

    # 日胜率
    win_rate = float((returns > 0).mean())

    return {
        "total_return": round(total_return, 4),
        "annual_return": round(ann_return, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "calmar": round(calmar, 3) if calmar == calmar else np.nan,
        "daily_win_rate": round(win_rate, 4),
    }
