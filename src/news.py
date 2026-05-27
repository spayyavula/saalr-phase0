"""News acquisition: ticker-tagged articles for SPY and its locked top-10 holdings.

Per ``pre-registration.md`` §3:

- Window: locked in ``SPEC.news_window_*``.
- Filter: articles tagged with SPY or any of ``SPEC.spy_top10_holdings``.
- Fields used: only ``title`` + ``description`` are fed downstream to FinBERT.
- Timestamp: ``published_utc`` is the time we treat news as publicly knowable.

The vendor's news endpoint accepts a single ticker per call; we fan out across
the 11 tickers, dedupe by article ``id`` (an article tagged with multiple
holdings will be returned by multiple calls), and return one row per article.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Iterable, Iterator

from polygon import RESTClient

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)


def _client() -> RESTClient:
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "MASSIVE_API_KEY (or POLYGON_API_KEY) is not set; see .env.example"
        )
    return RESTClient(api_key=key)


def _tickers() -> tuple[str, ...]:
    return (SPEC.options_symbol, *SPEC.spy_top10_holdings)


def _iter_news_for_ticker(
    client: RESTClient,
    ticker: str,
    start: date,
    end: date,
) -> Iterator[dict]:
    published_gte = datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat()
    published_lt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc).isoformat()
    logger.info("news pull: ticker=%s [%s, %s)", ticker, published_gte, published_lt)
    yield from client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=published_gte,
        published_utc_lt=published_lt,
        order="asc",
        sort="published_utc",
        limit=1000,
    )


def fetch_news(
    start: date,
    end: date,
    tickers: Iterable[str] | None = None,
    inter_call_sleep_seconds: float = 0.2,
):
    """Fetch news for ``tickers`` (default: SPY + locked top-10 holdings) over
    [start, end), dedupe by article id, and return a pandas DataFrame with:

    - ``id`` — vendor article id
    - ``timestamp`` — published_utc as a UTC datetime
    - ``title``
    - ``description``
    - ``tickers`` — list of tickers the vendor tagged on this article

    Only the locked ``SPEC.news_fields_used`` (title + description) feed
    downstream. ``tickers`` is retained for audit only.
    """
    import pandas as pd

    client = _client()
    seen: dict[str, dict] = {}
    for ticker in tickers or _tickers():
        for article in _iter_news_for_ticker(client, ticker, start, end):
            article_dict = (
                article if isinstance(article, dict) else article.__dict__
            )
            article_id = article_dict.get("id")
            if article_id is None or article_id in seen:
                continue
            seen[article_id] = {
                "id": article_id,
                "timestamp": pd.to_datetime(
                    article_dict.get("published_utc"), utc=True
                ),
                "title": article_dict.get("title", ""),
                "description": article_dict.get("description", ""),
                "tickers": list(article_dict.get("tickers", [])),
            }
        time.sleep(inter_call_sleep_seconds)

    if not seen:
        return pd.DataFrame(columns=["id", "timestamp", "title", "description", "tickers"])
    return (
        pd.DataFrame.from_records(list(seen.values()))
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
