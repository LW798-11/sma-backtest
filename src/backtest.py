"""回测模块：向量化回测，考虑交易成本。

核心逻辑：
- 仓位信号滞后一日生效（避免未来函数）
- 换手时按单边佣金 + 印花税扣费（简化处理）
"""
import pandas as pd

# 简化费率：佣金万 2.5（最低 5 元忽略），卖出印花税 0.05%（2023-08-28 起）
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.0005  # 仅卖出收取


def run_backtest(close: pd.Series, signal: pd.Series) -> pd.DataFrame:
    """输入收盘价与 0/1 持仓信号，返回逐日回测结果 DataFrame。"""
    ret = close.pct_change().fillna(0.0)

    # 信号滞后一日生效，避免使用当天收盘信息交易当天
    position = signal.shift(1).fillna(0.0)

    # 换手率（仓位变化绝对值）；卖出收印花税，买卖都收佣金
    turnover = position.diff().abs().fillna(0.0)
    sell_turnover = position.diff().clip(upper=0).abs().fillna(0.0)
    cost = turnover * COMMISSION_RATE + sell_turnover * STAMP_TAX_RATE

    strategy_ret = position * ret - cost
    equity = (1 + strategy_ret).cumprod()
    buy_hold_equity = (1 + ret).cumprod()

    result = pd.DataFrame(
        {
            "close": close,
            "signal": signal,
            "position": position,
            "strategy_ret": strategy_ret,
            "equity": equity,
            "buy_hold_equity": buy_hold_equity,
        }
    )
    return result
