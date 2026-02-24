import os
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TOP_COUNT = int(os.getenv("TOP_COUNT", "10"))
TRADE_IDEA_COUNT = int(os.getenv("TRADE_IDEA_COUNT", "5"))


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 symbols from Wikipedia with a User-Agent header."""
    import requests
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(SP500_WIKI_URL, headers=headers)
    tables = pd.read_html(response.text)
    df = tables[0]
    return [symbol.replace(".", "-") for symbol in df["Symbol"].tolist()]


def get_daily_movers(top_n: int = TOP_COUNT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return top gainers/losers and full movers dataframe by daily percentage change."""
    tickers = get_sp500_tickers()
    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            ticker_df = data[ticker][["Open", "Close", "Volume"]].dropna()
            if ticker_df.empty:
                continue
            latest = ticker_df.iloc[-1]
            open_price = float(latest["Open"])
            close_price = float(latest["Close"])
            if open_price <= 0 or close_price <= 0:
                continue

            pct_change = ((close_price - open_price) / open_price) * 100
            avg_volume = float(ticker_df["Volume"].tail(5).mean()) if len(ticker_df) > 0 else float(latest["Volume"])
            volume_ratio = float(latest["Volume"]) / avg_volume if avg_volume > 0 else 0.0

            rows.append(
                {
                    "Ticker": ticker,
                    "Open": open_price,
                    "Close": close_price,
                    "Change %": pct_change,
                    "Volume": int(latest["Volume"]),
                    "Avg 5D Vol": int(avg_volume),
                    "Vol Ratio": volume_ratio,
                }
            )
        except Exception:
            continue

    movers = pd.DataFrame(rows)
    if movers.empty:
        raise RuntimeError("No stock data returned from yfinance.")

    movers = movers.sort_values("Change %", ascending=False)
    top_gainers = movers.head(top_n).copy()
    top_losers = movers.tail(top_n).sort_values("Change %", ascending=True).copy()
    return top_gainers, top_losers, movers


def build_market_overview(movers: pd.DataFrame) -> str:
    up_count = int((movers["Change %"] > 0).sum())
    down_count = int((movers["Change %"] < 0).sum())
    unchanged = len(movers) - up_count - down_count
    avg_move = float(movers["Change %"].mean())
    high_volume_names = movers.sort_values("Vol Ratio", ascending=False).head(5)["Ticker"].tolist()

    tone = "risk-on" if avg_move > 0 else "risk-off"
    return (
        f"Market breadth: <b>{up_count}</b> advancers vs <b>{down_count}</b> decliners "
        f"(<b>{unchanged}</b> unchanged). Average daily move is <b>{avg_move:+.2f}%</b>, "
        f"suggesting a <b>{tone}</b> session. Most unusual volume appears in: "
        f"<b>{', '.join(high_volume_names)}</b>."
    )


def generate_trade_ideas(movers: pd.DataFrame, count: int = TRADE_IDEA_COUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate heuristic day-trade and swing-trade ideas with suggested limit levels."""
    day_candidates = movers.sort_values(["Vol Ratio", "Change %"], ascending=[False, False]).head(count).copy()
    day_candidates["Idea"] = "Day Trade (Momentum)"
    day_candidates["Buy Limit"] = day_candidates["Close"] * 0.995
    day_candidates["Sell Limit"] = day_candidates["Close"] * 1.01
    day_candidates["Stop"] = day_candidates["Close"] * 0.99

    swing_source = pd.concat(
        [movers.sort_values("Change %", ascending=False).head(count), movers.sort_values("Change %", ascending=True).head(count)]
    ).drop_duplicates(subset=["Ticker"])
    swing_candidates = swing_source.head(count).copy()
    swing_candidates["Idea"] = swing_candidates["Change %"].apply(
        lambda v: "Swing Trade (Trend Follow)" if v > 0 else "Swing Trade (Mean Reversion)"
    )
    swing_candidates["Buy Limit"] = swing_candidates["Close"] * swing_candidates["Change %"].apply(
        lambda v: 0.99 if v > 0 else 0.985
    )
    swing_candidates["Sell Limit"] = swing_candidates["Close"] * swing_candidates["Change %"].apply(
        lambda v: 1.03 if v > 0 else 1.015
    )
    swing_candidates["Stop"] = swing_candidates["Close"] * 0.97

    day_cols = ["Ticker", "Idea", "Close", "Change %", "Vol Ratio", "Buy Limit", "Sell Limit", "Stop"]
    return day_candidates[day_cols], swing_candidates[day_cols]


def get_news_and_politics_context(max_items: int = 8) -> list[dict[str, str]]:
    """Pull Yahoo-hosted news headlines and infer possible market impact themes."""
    watchlist = ["^GSPC", "SPY", "QQQ", "XLE", "XLK", "XLF"]
    snippets: list[dict[str, str]] = []
    seen: set[str] = set()

    for symbol in watchlist:
        try:
            ticker = yf.Ticker(symbol)
            for item in (ticker.news or [])[:5]:
                title = item.get("title", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                publisher = item.get("publisher", "Unknown")
                link = item.get("link", "")
                summary = infer_impact_from_title(title)
                snippets.append(
                    {
                        "Symbol": symbol,
                        "Headline": title,
                        "Publisher": publisher,
                        "Impact View": summary,
                        "Link": link,
                    }
                )
                if len(snippets) >= max_items:
                    return snippets
        except Exception:
            continue

    return snippets


def infer_impact_from_title(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["fed", "rates", "inflation", "cpi", "treasury", "powell"]):
        return "Macro/rates headline: may pressure growth stocks and indexes if yields rise."
    if any(k in t for k in ["election", "senate", "congress", "tariff", "sanction", "war", "geopolit"]):
        return "Political/geopolitical risk: can increase volatility and rotation into defensives."
    if any(k in t for k in ["oil", "opec", "energy"]):
        return "Energy-driven catalyst: may influence inflation expectations and sector leadership."
    if any(k in t for k in ["earnings", "guidance", "forecast", "revenue"]):
        return "Corporate catalyst: likely stock-specific momentum and sector spillover."
    return "General market catalyst: monitor pre-market futures and sector breadth for confirmation."


def _format_table(df: pd.DataFrame, title: str, table_class: str) -> str:
    formatted = df.copy()
    for col in ["Open", "Close", "Buy Limit", "Sell Limit", "Stop"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"${x:,.2f}")
    if "Change %" in formatted.columns:
        formatted["Change %"] = formatted["Change %"].map(lambda x: f"{x:+.2f}%")
    if "Volume" in formatted.columns:
        formatted["Volume"] = formatted["Volume"].map(lambda x: f"{x:,}")
    if "Avg 5D Vol" in formatted.columns:
        formatted["Avg 5D Vol"] = formatted["Avg 5D Vol"].map(lambda x: f"{x:,}")
    if "Vol Ratio" in formatted.columns:
        formatted["Vol Ratio"] = formatted["Vol Ratio"].map(lambda x: f"{x:.2f}x")

    html = formatted.to_html(index=False, classes=table_class, border=0, justify="center", escape=False)
    return f"<h2>{title}</h2>{html}"


def _format_news(news_items: list[dict[str, str]]) -> str:
    if not news_items:
        return "<p>No Yahoo Finance news items were available at runtime.</p>"

    rows = []
    for item in news_items:
        link = item["Link"]
        headline_html = f'<a href="{link}">{item["Headline"]}</a>' if link else item["Headline"]
        rows.append(
            {
                "Symbol": item["Symbol"],
                "Headline": headline_html,
                "Publisher": item["Publisher"],
                "Impact View": item["Impact View"],
            }
        )
    news_df = pd.DataFrame(rows)
    return _format_table(news_df, "Yahoo Finance News & Politics Watch", "stocks-table news")


def build_report_html(top_n: int = TOP_COUNT, idea_count: int = TRADE_IDEA_COUNT) -> str:
    gainers, losers, movers = get_daily_movers(top_n=top_n)
    market_overview = build_market_overview(movers)
    day_ideas, swing_ideas = generate_trade_ideas(movers, count=idea_count)
    news_html = _format_news(get_news_and_politics_context())
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    gainers_html = _format_table(gainers, f"Top {top_n} Gainers", "stocks-table gainers")
    losers_html = _format_table(losers, f"Top {top_n} Losers", "stocks-table losers")
    day_ideas_html = _format_table(day_ideas, f"Top {idea_count} Day-Trade Ideas", "stocks-table ideas")
    swing_ideas_html = _format_table(swing_ideas, f"Top {idea_count} Swing-Trade Ideas", "stocks-table ideas")

    return f"""
    <html>
    <head>
      <style>
        body {{ font-family: Arial, sans-serif; color: #1f2937; line-height: 1.45; }}
        h1 {{ margin-bottom: 4px; }}
        h2 {{ margin-top: 24px; color: #111827; }}
        .meta {{ color: #6b7280; margin-bottom: 20px; }}
        .overview {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; border-radius: 8px; margin-bottom: 12px; }}
        .disclaimer {{ font-size: 12px; color: #6b7280; margin-top: 16px; }}
        .stocks-table {{ border-collapse: collapse; width: 100%; max-width: 1000px; }}
        .stocks-table th, .stocks-table td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: center; }}
        .stocks-table th {{ background-color: #f3f4f6; }}
        .stocks-table a {{ color: #2563eb; text-decoration: none; }}
        .gainers td:nth-child(4) {{ color: #047857; font-weight: 600; }}
        .losers td:nth-child(4) {{ color: #b91c1c; font-weight: 600; }}
      </style>
    </head>
    <body>
      <h1>Daily S&P 500 Trading Report</h1>
      <p class="meta">Generated: {report_date}</p>
      <div class="overview"><b>Market Overview:</b> {market_overview}</div>
      {gainers_html}
      {losers_html}
      {day_ideas_html}
      {swing_ideas_html}
      {news_html}
      <p class="disclaimer">Trade ideas are heuristic and for informational purposes only, not financial advice.</p>
    </body>
    </html>
    """


if __name__ == "__main__":
    print(build_report_html())
