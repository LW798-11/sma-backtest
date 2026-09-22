"""数据加载模块：从 CSV 读取 iFinD 导出的行情数据，整理成收盘价宽表。"""
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "prices_raw.csv"


def load_close_prices(path: str | Path = DATA_FILE) -> pd.DataFrame:
    """读取行情 CSV，返回以日期为索引、各证券代码为列的收盘价 DataFrame。"""
    df = pd.read_csv(path, parse_dates=["time"])
    close = df.pivot(index="time", columns="thscode", values="close")
    close = close.sort_index()
    return close


if __name__ == "__main__":
    prices = load_close_prices()
    print(prices.head())
    print(f"\n数据区间: {prices.index[0].date()} ~ {prices.index[-1].date()}, 共 {len(prices)} 个交易日")
