"""FinBERT batch scoring for tagged news articles.

Per ``pre-registration.md`` §5 (locked in ``SPEC``):

- Model: ``SPEC.finbert_model_id`` (``ProsusAI/finbert``, 3-class).
- Input fields: ``SPEC.news_fields_used`` (``title``, ``description``)
  concatenated with a single space.
- Truncation: ``SPEC.finbert_max_tokens`` (512).
- Per-article score: ``SPEC.finbert_score_formula`` (``P_positive -
  P_negative``), range ``[-1, +1]``.

Imports of ``torch``/``transformers``/``pandas`` are deferred to call
time so ``python -m src.orchestrator status`` and similar control-plane
commands stay fast and so this module can be imported in environments
that don't yet have the heavy deps installed.

CLI::

    python -m src.sentiment --in data/train/news/2024-01.parquet \\
                            --out-source sentiment
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)


_model = None
_tokenizer = None


def _load_model():
    """Load FinBERT and its tokenizer; cache module-level for reuse."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("loading FinBERT %s ...", SPEC.finbert_model_id)
    tokenizer = AutoTokenizer.from_pretrained(SPEC.finbert_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(SPEC.finbert_model_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("FinBERT on CUDA")
    else:
        logger.info("FinBERT on CPU")

    _tokenizer = tokenizer
    _model = model
    return tokenizer, model


def _format_text(row) -> str:
    """Concatenate the locked ``SPEC.news_fields_used`` fields with single spaces."""
    parts: list[str] = []
    for field in SPEC.news_fields_used:
        val = row.get(field)
        if val:
            parts.append(str(val).strip())
    return " ".join(parts)


def _label_indices(model) -> dict[str, int]:
    """Map ``positive`` / ``neutral`` / ``negative`` -> output index for the
    loaded model. Defensive against id2label-ordering changes."""
    id2label = {int(i): str(label).lower() for i, label in model.config.id2label.items()}
    mapping: dict[str, int] = {}
    for idx, label in id2label.items():
        if "pos" in label:
            mapping["positive"] = idx
        elif "neg" in label:
            mapping["negative"] = idx
        elif "neu" in label:
            mapping["neutral"] = idx
    expected = {"positive", "negative", "neutral"}
    if set(mapping) != expected:
        raise RuntimeError(
            f"FinBERT id2label {id2label} does not map cleanly to {expected}; "
            f"resolved {mapping}"
        )
    return mapping


def score_articles(
    news_df,  # pandas.DataFrame
    batch_size: int = 32,
):
    """Score ``news_df`` with FinBERT. Returns a copy with added columns:

    - ``p_negative``, ``p_neutral``, ``p_positive`` — softmax probabilities
    - ``sentiment_score`` — ``p_positive - p_negative`` per
      ``SPEC.finbert_score_formula``

    All input columns (incl. ``timestamp``) are preserved unchanged. Input
    row order is preserved.
    """
    import torch

    if len(news_df) == 0:
        return news_df.assign(
            p_negative=[], p_neutral=[], p_positive=[], sentiment_score=[]
        )

    tokenizer, model = _load_model()
    label_idx = _label_indices(model)
    on_cuda = next(model.parameters()).is_cuda

    texts = [_format_text(row) for _, row in news_df.iterrows()]
    p_pos: list[float] = []
    p_neg: list[float] = []
    p_neu: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=SPEC.finbert_max_tokens,
                padding=True,
                return_tensors="pt",
            )
            if on_cuda:
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row_probs in probs:
                p_pos.append(float(row_probs[label_idx["positive"]]))
                p_neg.append(float(row_probs[label_idx["negative"]]))
                p_neu.append(float(row_probs[label_idx["neutral"]]))

    return news_df.assign(
        p_negative=p_neg,
        p_neutral=p_neu,
        p_positive=p_pos,
        sentiment_score=[pos - neg for pos, neg in zip(p_pos, p_neg)],
    )


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", "./data"))


def score_parquet(
    input_path: Path,
    out_source: str = "sentiment",
    data_root: Optional[Path] = None,
    batch_size: int = 32,
):
    """Read a news parquet, score it, and write the scored frame back
    through ``storage.write_partitioned_parquet`` so it lands in the same
    split as the source month and the MANIFEST gets a sha256 entry."""
    import pandas as pd

    from src import storage

    data_root = data_root or _data_root()
    logger.info("scoring %s", input_path)
    df = pd.read_parquet(input_path)
    if df.empty:
        logger.info("input empty; nothing to score")
        return []
    scored = score_articles(df, batch_size=batch_size)
    return storage.write_partitioned_parquet(
        scored, data_root, out_source, timestamp_col="timestamp"
    )


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="input_path", required=True, type=Path,
        help="Path to a news parquet file (one month).",
    )
    parser.add_argument(
        "--out-source", default="sentiment",
        help="Source name under data/{split}/{source}/. Default: sentiment.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="FinBERT batch size. Default: 32.",
    )
    args = parser.parse_args(argv)

    results = score_parquet(
        input_path=args.input_path,
        out_source=args.out_source,
        batch_size=args.batch_size,
    )
    for r in results:
        print(f"{r.path} rows={r.row_count} sha256={r.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
