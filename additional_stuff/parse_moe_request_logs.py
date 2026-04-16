#!/usr/bin/env python3
"""Parse and ingest MoE request profiling logs without loading all data in RAM.

The logs are per-request `.msgpack.zlib` files. This module streams files,
extracts top-k expert activations per token/layer, and stores flattened rows in
SQLite for out-of-core analysis.
"""

from __future__ import annotations

import argparse
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import msgspec
import numpy as np


DEFAULT_SPLITS: tuple[str, ...] = ("prefill", "decode")
SPLIT_TO_ID: dict[str, int] = {"prefill": 0, "decode": 1}
ID_TO_SPLIT: dict[int, str] = {0: "prefill", 1: "decode"}


@dataclass
class IngestSummary:
    files_scanned: int
    files_ingested: int
    rows_inserted: int
    layer0_embedding_rows: int
    files_failed: int


def split_name_to_id(split: str) -> int:
    split_id = SPLIT_TO_ID.get(split)
    if split_id is None:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of: {sorted(SPLIT_TO_ID)}"
        )
    return split_id


def split_id_to_name(split_id: int) -> str:
    return ID_TO_SPLIT.get(int(split_id), f"split_{split_id}")


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required for DataFrame output. "
            "Install with: pip install pandas"
        ) from exc
    return pd


def iter_profile_files(
    log_root: str | Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
    max_files: int | None = None,
    include_tmp: bool = False,
) -> Iterator[tuple[str, Path]]:
    """Yield `(split, path)` for `.msgpack.zlib` files under split folders."""
    root = Path(log_root)
    yielded = 0

    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue

        for file_path in sorted(split_dir.iterdir()):
            name = file_path.name
            if name.endswith(".msgpack.zlib"):
                pass
            elif include_tmp and name.endswith(".msgpack.zlib.tmp"):
                pass
            else:
                continue

            yield split, file_path
            yielded += 1
            if max_files is not None and yielded >= max_files:
                return


def decode_profile_file(file_path: str | Path) -> dict[str, Any]:
    """Decode a single `.msgpack.zlib` request profile file."""
    path = Path(file_path)
    payload = path.read_bytes()
    decoded = msgspec.msgpack.decode(zlib.decompress(payload))
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(decoded)}")
    return decoded


def _extract_topk_rows_from_record(
    *,
    path: str | Path,
    split_id: int,
    record: dict[str, Any],
    top_k: int,
) -> tuple[list[tuple[Any, ...]], tuple[Any, ...], list[tuple[Any, ...]]]:
    rows: list[tuple[Any, ...]] = []
    layer0_rows: list[tuple[Any, ...]] = []

    request_id = record.get("request_id", "")
    iterations = record.get("moe_expert_activation", []) or []

    for iteration in iterations:
        iter_no = int(iteration.get("iter_no", -1))
        layers = iteration.get("layers", []) or []

        # Store routed_layer0_input_embeddings in compressed form.
        layer0_input = np.asarray(
            iteration.get("layer0_input_embedding", []), dtype=np.float16
        )
        if layer0_input.ndim == 2 and layer0_input.size > 0 and split_id == 1:
            if layer0_input.shape[0] != 1:
                print(f"Expected batch size of 1 for layer0_input_embedding in {path} but got {layer0_input.shape[0]} in request_id={request_id}, iter_no={iter_no}")
                continue
            zlib_blob = zlib.compress(layer0_input.tobytes(), level=1)
            layer0_rows.append(
                (
                    split_id,
                    request_id,
                    iter_no,
                    int(layer0_input.shape[0]),
                    int(layer0_input.shape[1]),
                    "float16",
                    zlib_blob,
                )
            )

        for layer in layers:
            layer_no = int(layer.get("layer_no", -1))
            probs = np.asarray(layer.get("expert_probabilities", []),
                               dtype=np.float32)
            if probs.ndim != 2 or probs.size == 0:
                continue

            num_tokens, num_experts = probs.shape
            if num_tokens == 0 or num_experts == 0:
                continue

            k = min(top_k, num_experts)
            if k <= 0:
                continue

            if k == num_experts:
                topk_idx = np.argsort(-probs, axis=1)[:, :k]
            else:
                partial_idx = np.argpartition(probs, -k, axis=1)[:, -k:]
                partial_vals = np.take_along_axis(probs, partial_idx, axis=1)
                order = np.argsort(-partial_vals, axis=1)
                topk_idx = np.take_along_axis(partial_idx, order, axis=1)

            topk_vals = np.take_along_axis(probs, topk_idx, axis=1)

            token_idx = np.repeat(np.arange(num_tokens, dtype=np.int32), k)

            expert_flat = topk_idx.reshape(-1)
            prob_flat = topk_vals.reshape(-1)

            for tok, expert, prob in zip(
                token_idx,
                expert_flat,
                prob_flat,
                strict=False,
            ):
                rows.append((
                    split_id,
                    request_id,
                    iter_no,
                    layer_no,
                    int(tok),
                    int(expert),
                    float(prob),
                ))

    request_row = (
        split_id,
        request_id,
        len(iterations),
    )
    return rows, request_row, layer0_rows


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-200000;

        CREATE TABLE IF NOT EXISTS request_summary (
            split_id INTEGER NOT NULL CHECK (split_id IN (0, 1)),
            request_id TEXT NOT NULL,
            num_iterations INTEGER NOT NULL,
            PRIMARY KEY (split_id, request_id)
        );

        CREATE TABLE IF NOT EXISTS layer_token_topk (
            split_id INTEGER NOT NULL CHECK (split_id IN (0, 1)),
            request_id TEXT NOT NULL,
            iter_no INTEGER NOT NULL,
            layer_no INTEGER NOT NULL,
            token_idx INTEGER NOT NULL,
            expert_id INTEGER NOT NULL,
            probability REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS layer0_input_embeddings (
            split_id INTEGER NOT NULL CHECK (split_id IN (0, 1)),
            request_id TEXT NOT NULL,
            iter_no INTEGER NOT NULL,
            num_tokens INTEGER NOT NULL,
            hidden_size INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            zlib_blob BLOB NOT NULL,
            PRIMARY KEY (split_id, request_id, iter_no)
        );

        CREATE INDEX IF NOT EXISTS idx_topk_split_layer
            ON layer_token_topk (split_id, layer_no);

        CREATE INDEX IF NOT EXISTS idx_topk_split_expert
            ON layer_token_topk (split_id, expert_id);

        CREATE INDEX IF NOT EXISTS idx_topk_request
            ON layer_token_topk (request_id, iter_no, layer_no);

        CREATE INDEX IF NOT EXISTS idx_layer0_request
            ON layer0_input_embeddings (request_id, iter_no);
        """
    )


def ingest_topk_rows_to_sqlite(
    *,
    log_root: str | Path,
    sqlite_path: str | Path,
    top_k: int = 2,
    splits: Sequence[str] = DEFAULT_SPLITS,
    max_files: int | None = None,
    batch_size: int = 50_000,
    progress_every_files: int = 20,
) -> IngestSummary:
    """Ingest top-k rows into SQLite, suitable for very large datasets."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    sqlite_file = Path(sqlite_path)
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(sqlite_file))
    _create_schema(conn)

    summary_rows = []
    topk_rows_batch: list[tuple[Any, ...]] = []
    layer0_rows_batch: list[tuple[Any, ...]] = []

    files_scanned = 0
    files_ingested = 0
    rows_inserted = 0
    layer0_embedding_rows = 0
    files_failed = 0

    try:
        for split, path in iter_profile_files(
            log_root=log_root,
            splits=splits,
            max_files=max_files,
        ):
            files_scanned += 1
            try:
                split_id = split_name_to_id(split)
                record = decode_profile_file(path)
                rows, request_row, layer0_rows = _extract_topk_rows_from_record(
                    path=path,
                    split_id=split_id,
                    record=record,
                    top_k=top_k,
                )

                summary_rows.append(request_row)
                topk_rows_batch.extend(rows)
                layer0_rows_batch.extend(layer0_rows)
                rows_inserted += len(rows)
                layer0_embedding_rows += len(layer0_rows)
                files_ingested += 1
            except Exception as exc:  # pragma: no cover
                files_failed += 1
                print(f"[WARN] Failed to parse {path}: {exc}")

            if len(topk_rows_batch) >= batch_size:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO request_summary
                    (split_id, request_id, num_iterations)
                    VALUES (?, ?, ?)
                    """,
                    summary_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO layer_token_topk
                    (split_id, request_id, iter_no, layer_no,
                     token_idx, expert_id, probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    topk_rows_batch,
                )
                if layer0_rows_batch:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO layer0_input_embeddings
                        (split_id, request_id, iter_no, num_tokens, hidden_size,
                         dtype, zlib_blob)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        layer0_rows_batch,
                    )
                conn.commit()
                summary_rows.clear()
                topk_rows_batch.clear()
                layer0_rows_batch.clear()

            if progress_every_files and files_scanned % progress_every_files == 0:
                print(
                    "[INFO] scanned=%d ingested=%d rows=%d failed=%d"
                    % (files_scanned, files_ingested, rows_inserted, files_failed)
                )

        if summary_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO request_summary
                (split_id, request_id, num_iterations)
                VALUES (?, ?, ?)
                """,
                summary_rows,
            )
        if topk_rows_batch:
            conn.executemany(
                """
                INSERT INTO layer_token_topk
                (split_id, request_id, iter_no, layer_no,
                 token_idx, expert_id, probability)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                topk_rows_batch,
            )
        if layer0_rows_batch:
            conn.executemany(
                """
                INSERT OR REPLACE INTO layer0_input_embeddings
                (split_id, request_id, iter_no, num_tokens, hidden_size,
                 dtype, zlib_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                layer0_rows_batch,
            )
        conn.commit()
    finally:
        conn.close()

    return IngestSummary(
        files_scanned=files_scanned,
        files_ingested=files_ingested,
        rows_inserted=rows_inserted,
        layer0_embedding_rows=layer0_embedding_rows,
        files_failed=files_failed,
    )


def iter_sql_as_dataframes(
    *,
    sqlite_path: str | Path,
    sql: str,
    params: Sequence[Any] | None = None,
    chunksize: int = 200_000,
) -> Iterator[Any]:
    """Yield query results as DataFrame chunks to keep RAM usage bounded."""
    pd = _require_pandas()
    conn = sqlite3.connect(str(sqlite_path))
    try:
        query_iter = pd.read_sql_query(
            sql,
            conn,
            params=tuple(params or ()),
            chunksize=chunksize,
        )
        yield from query_iter
    finally:
        conn.close()


def read_sql_as_dataframe(
    *,
    sqlite_path: str | Path,
    sql: str,
    params: Sequence[Any] | None = None,
) -> Any:
    """Read a SQL query result fully into a DataFrame.

    Use this only for already-aggregated queries that are expected to be small.
    """
    pd = _require_pandas()
    conn = sqlite3.connect(str(sqlite_path))
    try:
        return pd.read_sql_query(sql, conn, params=tuple(params or ()))
    finally:
        conn.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    guide = """
Quick guide (ingest):
  --log-root:    Root directory containing split folders.
                 Expected structure includes subdirs like 'prefill' and 'decode'.
  --sqlite-path: Output SQLite file path to create/update.
  --top-k:       Number of top experts to keep per token/layer.
  --splits:      Split folders to parse (stored as split_id: prefill=0, decode=1).
  --max-files:   Optional cap on number of files for smoke tests.
  --batch-size:  Number of flattened rows per DB commit.

Examples:
  Full ingest:
    python parse_moe_request_logs.py ingest \
      --log-root /data/logs/qwen1.5_2.7B \
      --sqlite-path ./artifacts/moe_logs.sqlite \
      --top-k 2 --splits prefill decode --batch-size 50000

  Schema-only DB creation (no rows):
    mkdir -p /tmp/moe_schema/prefill /tmp/moe_schema/decode
    python parse_moe_request_logs.py ingest \
      --log-root /tmp/moe_schema \
      --sqlite-path ./artifacts/moe_schema_only.sqlite
"""

    parser = argparse.ArgumentParser(
        description="Parse MoE request logs into SQLite for out-of-core analysis.",
        epilog=guide,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest logs into SQLite")
    ingest.add_argument(
        "--log-root",
        required=True,
        help=(
            "Root log directory containing split folders "
            "(typically prefill/decode)."
        ),
    )
    ingest.add_argument(
        "--sqlite-path",
        required=True,
        help="Output SQLite path to create/update.",
    )
    ingest.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k experts to keep per token/layer (default: 2).",
    )
    ingest.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split subdirectories to parse (default: prefill decode).",
    )
    ingest.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file-count limit for smoke runs.",
    )
    ingest.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per transaction commit (default: 50000).",
    )

    query = subparsers.add_parser("query", help="Run SQL and print sample rows")
    query.add_argument(
        "--sqlite-path",
        required=True,
        help="SQLite DB path to query.",
    )
    query.add_argument(
        "--sql",
        required=True,
        help="SQL string to execute.",
    )
    query.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="DataFrame chunk size for streaming query output.",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        summary = ingest_topk_rows_to_sqlite(
            log_root=args.log_root,
            sqlite_path=args.sqlite_path,
            top_k=args.top_k,
            splits=args.splits,
            max_files=args.max_files,
            batch_size=args.batch_size,
        )
        print(summary)
        return

    if args.command == "query":
        chunks = iter_sql_as_dataframes(
            sqlite_path=args.sqlite_path,
            sql=args.sql,
            chunksize=args.chunksize,
        )
        total_rows = 0
        for idx, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"[chunk={idx}] rows={len(chunk)}")
            print(chunk.head(5).to_string(index=False))
        print(f"total_rows={total_rows}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()