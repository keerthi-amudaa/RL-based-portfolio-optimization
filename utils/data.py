import yfinance as yf
import pandas as pd


ASSET_CATEGORIES = {
    "US Tech Stocks": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "EV & Energy": ["TSLA", "NIO", "XPEV", "FSLR"],
    "Finance": ["JPM", "GS", "BAC", "MS"],
    "ETFs": ["SPY", "QQQ", "ARKK", "VTI"],
    "Crypto Proxies": ["COIN", "MSTR"],
}


def fetch_prices(tickers, period="1y"):
    data = yf.download(tickers, period=period)["Close"]
    return data.dropna()
