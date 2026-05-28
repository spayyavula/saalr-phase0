"""CLI entrypoint for the Phase 0 acquisition layer.

Examples
--------
::

    python -m src.data_acquisition risk-free --start 2024-01-01 --end 2024-12-31
    python -m src.data_acquisition underlying --start 2024-01-01 --end 2024-01-31
    python -m src.data_acquisition news --start 2024-01-01 --end 2024-01-31
    python -m src.data_acquisition all --start 2024-01-01 --end 2024-12-31

Writes parquet files under ``data/{split}/{source}/YYYY-MM.parquet`` and
updates ``data/MANIFEST.json``. The split is derived from each row's date
against the locked windows in ``locked_spec.SPEC``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed yet; .env still works if env exported manually
    pass

from src import news, options, risk_free, storage, underlying

logger = logging.getLogger(__name__)


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", "./data"))


def _write(df, source: str) -> None:
    results = storage.write_partitioned_parquet(df, _data_root(), source)
    for r in results:
        print(f"{r.path} rows={r.row_count} sha256={r.sha256[:12]}")


def cmd_risk_free(args: argparse.Namespace) -> None:
    df = risk_free.fetch_risk_free(args.start, args.end)
    _write(df, "risk_free")


def cmd_underlying(args: argparse.Namespace) -> None:
    df = underlying.fetch_underlying(args.start, args.end)
    _write(df, "underlying")


def cmd_news(args: argparse.Namespace) -> None:
    df = news.fetch_news(args.start, args.end)
    _write(df, "news")


def cmd_options(args: argparse.Namespace) -> None:
    from datetime import timedelta

    data_root = _data_root()
    cursor = args.start.replace(day=1)
    while cursor <= args.end:
        month = cursor.strftime("%Y-%m")
        underlying_df = storage.read_month_across_splits(data_root, "underlying", month)
        if underlying_df.empty:
            print(f"{month}: no underlying parquet found across splits; skipping")
        else:
            df = options.fetch_contracts_for_month(month, underlying_df)
            if len(df) > 0:
                _write(df, "options")
            else:
                print(f"{month}: no contracts returned")
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)


def cmd_all(args: argparse.Namespace) -> None:
    cmd_risk_free(args)
    cmd_underlying(args)
    cmd_news(args)
    cmd_options(args)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in [
        ("risk-free", cmd_risk_free),
        ("underlying", cmd_underlying),
        ("news", cmd_news),
        ("options", cmd_options),
        ("all", cmd_all),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--start", type=_parse_date, required=True)
        p.add_argument("--end", type=_parse_date, required=True)
        p.set_defaults(func=handler)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
