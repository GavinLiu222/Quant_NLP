"""FinBERT representations for the tokenized news pool.

Reads already-encoded ids/masks -- the `<text>_tokens` / `<text>_mask` columns,
whichever text column they came from -- runs them through FinBERT and writes
five sentence-level representations plus the
fine-tuned 3-way head's raw scores per article to one parquet file, alongside
the original `symbols` and `created_at`:

    cls_last          [CLS] of the last encoder layer (layer 12)
    pooler            tanh(W . cls_last + b), the fine-tuned pooling head
    mean_last         mean of the last layer over non-[PAD] tokens
    cls_second_last   [CLS] of the second-to-last encoder layer (layer 11)
    mean_second_last  mean of the second-to-last layer over non-[PAD] tokens
    logits            the classification head applied to `pooler`, pre-softmax

The five representations are 768-dimensional float32; `logits` is 3-dimensional
float32, in ProsusAI's own label order -- positive, negative, neutral, which is
NOT the negative/neutral/positive most sentiment checkpoints use.  That order is
read off the checkpoint rather than assumed, and is stored in the parquet
schema metadata under `logit_labels` so a consumer never has to guess.  Softmax
is deliberately left to the consumer: it throws away the overall scale, and a
tree model may well prefer the raw margins.

Rows whose `symbols`, `created_at`, ids or mask is missing are dropped, so the
output is shorter than the input CSV and the two can no longer be lined up by
position -- join on `symbols` + `created_at`, and see the note in the chunk
loop for where the blanks come from.

Two things keep this from taking hours.  Every input row is padded to a full 512
tokens, but the median article is 24 tokens long -- so within
each chunk rows are sorted by true length, batched, and trimmed to the longest
row in the batch.  Trailing [PAD] is masked out of attention anyway, so dropping
it changes nothing but the compute, which falls ~4.7x.  Batch size then follows
a token budget rather than a row count, keeping short-sequence batches large.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

import config

# --------------------------------------------------------------------------
# Config -- everything tunable lives here; paths and device come from config.py.
# --------------------------------------------------------------------------
TOKENIZED_CSV = config.NEWS_TOKENIZED_CSV
OUTPUT_PARQUET = config.FINBERT_PARQUET

# CSV rows parsed and held in memory at once; also one parquet row group.
# 20k rows of output is ~300 MB, which the writer flushes straight to disk.
CHUNK_ROWS = 20_000

# Tokens per forward pass (rows x padded width).  With length-sorted batches
# this is a far better throttle than a fixed row count: ~680 rows/batch at the
# median width of 24, dropping to 32 rows for the 512-token articles.
TOKENS_PER_BATCH = 16_384
MAX_BATCH_ROWS = 512

# Embeddings are near-incompressible, so this buys ~5%; it costs little.
COMPRESSION = "zstd"

# None = the whole file.  Set to e.g. 2_000 for a smoke test.
LIMIT_ROWS = None

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
DEVICE = config.DEVICE if config.DEVICE != "auto" else (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")

# Load the sequence-classification checkpoint and split it into the two pieces
# this script uses: the `.bert` trunk, which carries the fine-tuned pooler
# weights, and the `.classifier` head on top of it.  Loading the bare
# `AutoModel` would give the trunk too, but would leave it ambiguous whether the
# pooler came from the checkpoint or from a fresh init -- and would not give the
# head at all.
_classifier = AutoModelForSequenceClassification.from_pretrained(
    str(config.MODEL_DIR), local_files_only=True)
_classifier.eval()
_classifier.to(DEVICE)
model = _classifier.bert
head = _classifier.classifier

HIDDEN = model.config.hidden_size
REPRESENTATIONS = ["cls_last", "pooler", "mean_last",
                   "cls_second_last", "mean_second_last"]
LOGITS = "logits"

# positive / negative / neutral for this checkpoint -- taken from the config so
# that a different fine-tune cannot silently permute the columns.
LABELS = [str(_classifier.config.id2label[i]).lower()
          for i in range(_classifier.config.num_labels)]

# Every output column is a fixed-size float32 list; only the widths differ.
WIDTHS = {name: HIDDEN for name in REPRESENTATIONS} | {LOGITS: len(LABELS)}
OUTPUTS = REPRESENTATIONS + [LOGITS]

schema = pa.schema(
    [("symbols", pa.string()), ("created_at", pa.timestamp("us", tz="UTC"))]
    + [(name, pa.list_(pa.float32(), WIDTHS[name])) for name in OUTPUTS],
    metadata={"logit_labels": ",".join(LABELS)},
)


def parse_int_lists(column: pd.Series) -> np.ndarray:
    """Turn a column of "[101, 2054, ...]" strings into one 2-D int array.

    `to_csv` stringified the lists on the way out; `np.fromstring` parses them
    back at C speed, ~0.4 s per 20k rows.  Every row is the same width because
    the tokenizer padded to `max_length`, so they stack directly.
    """
    return np.stack([np.fromstring(s[1:-1], dtype=np.int64, sep=",")
                     for s in column])


def encode(ids: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Run one already-trimmed batch and pull every output column out."""
    ids = torch.from_numpy(ids).to(DEVICE)
    mask = torch.from_numpy(mask).to(DEVICE)

    with torch.inference_mode():
        # `token_type_ids` is omitted deliberately: these are single sequences,
        # so BERT's all-zeros default is exactly what the tokenizer produced.
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)

        # What `BertForSequenceClassification.forward` would return for this
        # batch: the head reads the pooler output, and the dropout between them
        # is the identity in eval mode.  Reusing the trunk's output this way
        # makes the logits free -- no second forward pass over the text.  It has
        # to happen inside `inference_mode`, because the head's weights still
        # carry `requires_grad` and autograd refuses to touch inference tensors.
        logits = head(out.pooler_output)

    # hidden_states is 13 tensors for BERT-base: [0] is the embedding output
    # and [1:] the twelve encoder layers, so [-1] is layer 12 and [-2] layer 11.
    last, second_last = out.hidden_states[-1], out.hidden_states[-2]

    # Mean over real tokens only -- [PAD] positions carry a nonzero vector and
    # would otherwise drag every short article toward the same point.  [CLS]
    # and [SEP] are kept, which is the usual reading of "masked mean".
    weights = mask.unsqueeze(-1).to(last.dtype)
    n_real = weights.sum(dim=1).clamp(min=1.0)

    batch = {
        "cls_last": last[:, 0],
        "pooler": out.pooler_output,
        "mean_last": (last * weights).sum(dim=1) / n_real,
        "cls_second_last": second_last[:, 0],
        "mean_second_last": (second_last * weights).sum(dim=1) / n_real,
        LOGITS: logits,
    }
    return {k: v.float().cpu().numpy() for k, v in batch.items()}


def encode_chunk(ids: np.ndarray, mask: np.ndarray,
                 progress: tqdm) -> dict[str, np.ndarray]:
    """Encode a chunk, returning every output back in the input row order."""
    lengths = mask.sum(axis=1)
    order = np.argsort(lengths, kind="stable")      # short articles first
    out = {name: np.empty((len(order), WIDTHS[name]), dtype=np.float32)
           for name in OUTPUTS}

    start = 0
    while start < len(order):
        # Sorted ascending, so the last row taken sets the padded width.  Shrink
        # the batch until rows x width fits the token budget.
        rows = min(MAX_BATCH_ROWS, len(order) - start)
        while rows > 1 and rows * int(lengths[order[start + rows - 1]]) > TOKENS_PER_BATCH:
            rows = max(1, min(rows - 1,
                              TOKENS_PER_BATCH // int(lengths[order[start + rows - 1]])))

        rows_idx = order[start:start + rows]
        width = int(lengths[rows_idx].max())
        for name, values in encode(ids[rows_idx, :width], mask[rows_idx, :width]).items():
            out[name][rows_idx] = values                # scatter back into place

        start += rows
        progress.update(rows)

    return out


def to_arrow(values: np.ndarray) -> pa.FixedSizeListArray:
    """One (n, width) float32 block as a parquet fixed-size list column."""
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.ascontiguousarray(values).reshape(-1)), values.shape[1])


# The two list columns are named after whichever text column was encoded
# (`<text>_tokens` / `<text>_mask`), so read that name off the header rather
# than hard-coding it.  Unpacking fails loudly unless exactly one matches.
(TOKENS_COLUMN,) = [c for c in pd.read_csv(TOKENIZED_CSV, nrows=0).columns
                    if c.endswith("_tokens")]
MASK_COLUMN = TOKENS_COLUMN.removesuffix("_tokens") + "_mask"

# Only used to size the progress bar, so a miscount would be harmless: no field
# in this file (token lists, tickers, ISO timestamps) can contain a newline.
with open(TOKENIZED_CSV, "rb") as handle:
    total_rows = sum(1 for _ in handle) - 1
if LIMIT_ROWS is not None:
    total_rows = min(total_rows, LIMIT_ROWS)

print(f"FinBERT on {DEVICE}: {total_rows:,} articles ({TOKENS_COLUMN}) -> {OUTPUT_PARQUET.name}")

OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
reader = pd.read_csv(TOKENIZED_CSV, chunksize=CHUNK_ROWS, nrows=LIMIT_ROWS,
                     dtype=str)   # the list columns arrive as strings either way
written = 0
dropped = 0
with pq.ParquetWriter(OUTPUT_PARQUET, schema, compression=COMPRESSION) as writer:
    with tqdm(total=total_rows, unit="doc", smoothing=0.05) as progress:
        for chunk in reader:
            # Drop any row with a missing field.  33 articles in the current
            # pool carry no ticker at all: step 1's `~symbols.str.contains(',')`
            # was meant to drop multi-ticker rows, but under pandas 3
            # `str.contains` returns False for a missing value, so `~` keeps it.
            # An article with no ticker cannot be joined to a price series, and
            # a missing id/mask cannot be encoded, so neither is worth a row.
            kept = chunk.dropna()
            dropped += len(chunk) - len(kept)
            progress.update(len(chunk) - len(kept))
            if kept.empty:
                continue

            ids = parse_int_lists(kept[TOKENS_COLUMN])
            mask = parse_int_lists(kept[MASK_COLUMN])

            encoded = encode_chunk(ids, mask, progress)
            writer.write_table(pa.Table.from_arrays(
                [pa.array(kept["symbols"].to_numpy(), type=pa.string()),
                 pa.array(pd.to_datetime(kept["created_at"], utc=True).to_numpy(),
                          type=pa.timestamp("us", tz="UTC"))]
                + [to_arrow(encoded[name]) for name in OUTPUTS],
                schema=schema))
            written += len(kept)

print(f"Wrote {written:,} rows x ({len(REPRESENTATIONS)} x {HIDDEN}d + "
      f"{len(LABELS)}d logits: {'/'.join(LABELS)}) to {OUTPUT_PARQUET}")
if dropped:
    print(f"dropped {dropped:,} row(s) with a missing field")
