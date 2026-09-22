"""主程序：加载数据 -> 生成信号 -> 回测 -> 计算指标 -> 画图。

运行：python main.py
输出：results/backtest.png、results/metrics.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import load_close_prices
from src.metrics import performance_summary
from src.strategy import sma_crossover_signal

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STOCK = "600519.SH"  # 贵州茅台
BENCHMARK = "000300.SH"  # 沪深300
SHORT_WINDOW = 20
LONG_WINDOW = 60


def plot_backtest(bt: pd.DataFrame, prices: pd.DataFrame, stock: str, benchmark: str) -> None:
    close = bt["close"]
    sma_short = close.rolling(SHORT_WINDOW).mean()
    sma_long = close.rolling(LONG_WINDOW).mean()

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1]},
    )

    # 1) 价格与均线 + 买卖点
    axes[0].plot(close.index, close, label=stock, color="gray", linewidth=1.0)
    axes[0].plot(close.index, sma_short, label=f"SMA{SHORT_WINDOW}", linewidth=1.0)
    axes[0].plot(close.index, sma_long, label=f"SMA{LONG_WINDOW}", linewidth=1.0)
    buy = bt[(bt["signal"] == 1) & (bt["signal"].shift(1) == 0)]
    sell = bt[(bt["signal"] == 0) & (bt["signal"].shift(1) == 1)]
    axes[0].scatter(buy.index, close[buy.index], marker="^", color="red", s=60, label="Buy", zorder=5)
    axes[0].scatter(sell.index, close[sell.index], marker="v", color="green", s=60, label="Sell", zorder=5)
    axes[0].set_title("Dual Moving Average Crossover Signals")
    axes[0].legend(loc="upper left")

    # 2) 净值曲线对比：策略 vs 买入持有 vs 基准
    axes[1].plot(bt.index, bt["equity"], label="Strategy", linewidth=1.4)
    axes[1].plot(bt.index, bt["buy_hold_equity"], label="Buy & Hold", linewidth=1.2, alpha=0.8)
    bench_norm = prices[benchmark] / prices[benchmark].iloc[0]
    axes[1].plot(bench_norm.index, bench_norm, label=benchmark, linewidth=1.2, alpha=0.8)
    axes[1].set_title("Equity Curve (initial = 1)")
    axes[1].legend(loc="upper left")

    # 3) 回撤
    peak = bt["equity"].cummax()
    drawdown = bt["equity"] / peak - 1.0
    axes[2].fill_between(drawdown.index, drawdown, 0, color="tomato", alpha=0.6)
    axes[2].set_title("Strategy Drawdown")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "backtest.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    prices = load_close_prices()

    close = prices[STOCK]
    signal = sma_crossover_signal(close, SHORT_WINDOW, LONG_WINDOW)
    bt = run_backtest(close, signal)

    # 计算策略、买入持有、基准的绩效指标
    rows = {
        "strategy": performance_summary(bt["strategy_ret"], bt["equity"]),
        "buy_and_hold": performance_summary(close.pct_change().fillna(0.0), bt["buy_hold_equity"]),
    }
    bench_ret = prices[BENCHMARK].pct_change().fillna(0.0)
    bench_equity = (1 + bench_ret).cumprod()
    rows["benchmark(000300.SH)"] = performance_summary(bench_ret, bench_equity)

    metrics = pd.DataFrame(rows).T
    metrics.to_csv(RESULTS_DIR / "metrics.csv", encoding="utf-8-sig")
    print(metrics)

    plot_backtest(bt, prices, STOCK, BENCHMARK)
    print(f"\n回测图表已保存: {RESULTS_DIR / 'backtest.png'}")


if __name__ == "__main__":
    main()
