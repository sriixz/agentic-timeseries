import yfinance as yf


def fetch_stock_data(symbol, period="1mo"):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)

        if history.empty:
            raise ValueError(
                f"No data returned for symbol '{symbol}' "
                f"with period '{period}'."
            )

        observations = []

        for date, row in history.iterrows():
            observations.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2)
            })

        return {
            "symbol": symbol,
            "period": period,
            "observations": observations
        }

    except Exception as error:
        raise RuntimeError(
            f"Failed to fetch stock data for {symbol}: {error}"
        ) from error