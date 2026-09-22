"""金融数据分析（EDA）：对标的和基准做描述性统计与可视化。

运行：python analysis.py
输出：results/analysis.png、results/descriptive_stats.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_close_prices

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STOCK = "600519.SH"  # 贵州茅台
BENCHMARK = "000300.SH"  # 沪深300


def descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """对日收益率做描述性统计。"""
    stats = returns.describe().T
    stats["annual_volatility"] = returns.std() * (252**0.5)
    stats["skew"] = returns.skew()
    stats["kurtosis"] = returns.kurtosis()
    return stats.round(4)


def plot_analysis(prices: pd.DataFrame, stock: str, benchmark: str) -> None:
    stock_ret = prices[stock].pct_change().dropna()
    bench_ret = prices[benchmark].pct_change().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # 1) 归一化价格走势
    norm = prices[[stock, benchmark]] / prices[[stock, benchmark]].iloc[0]
    axes[0, 0].plot(norm.index, norm[stock], label=stock, linewidth=1.2)
    axes[0, 0].plot(norm.index, norm[benchmark], label=benchmark, linewidth=1.2)
    axes[0, 0].set_title("Normalized Price")
    axes[0, 0].legend()

    # 2) 日收益率分布直方图
    axes[0, 1].hist(stock_ret, bins=60, alpha=0.7, label=stock)
    axes[0, 1].hist(bench_ret, bins=60, alpha=0.7, label=benchmark)
    axes[0, 1].set_title("Daily Returns Distribution")
    axes[0, 1].legend()

    # 3) 滚动 60 日年化波动率
    axes[1, 0].plot(stock_ret.index, stock_ret.rolling(60).std() * (252**0.5), label=stock, linewidth=1.2)
    axes[1, 0].plot(bench_ret.index, bench_ret.rolling(60).std() * (252**0.5), label=benchmark, linewidth=1.2)
    axes[1, 0].set_title("Rolling 60D Annualized Volatility")
    axes[1, 0].legend()

    # 4) 标的 vs 基准日收益散点（观察相关性 / Beta）
    aligned = pd.concat([stock_ret, bench_ret], axis=1, keys=["stock", "bench"]).dropna()
    axes[1, 1].scatter(aligned["bench"], aligned["stock"], s=8, alpha=0.4)
    beta = aligned["stock"].cov(aligned["bench"]) / aligned["bench"].var()
    corr = aligned["stock"].corr(aligned["bench"])
    axes[1, 1].set_title(f"Stock vs Benchmark (corr={corr:.2f}, beta={beta:.2f})")
    axes[1, 1].set_xlabel("Benchmark daily return")
    axes[1, 1].set_ylabel("Stock daily return")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    prices = load_close_prices()
    returns = prices.pct_change().dropna()

    stats = descriptive_stats(returns[[STOCK, BENCHMARK]])
    stats.to_csv(RESULTS_DIR / "descriptive_stats.csv", encoding="utf-8-sig")
    print(stats)

    plot_analysis(prices, STOCK, BENCHMARK)
    print(f"\n图表已保存: {RESULTS_DIR / 'analysis.png'}")
