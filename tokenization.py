"""Tokenize the cleaned news text with the local FinBERT tokenizer.

Reads the cleaned news, keeps only the text, ticker and timestamp, and encodes
the CONTENT_COLUMNS text column (e.g. `summary_filtered` or `body_filtered`)
into fixed-length 512-token id and attention-mask sequences.

Input
  config.NEWS_FILTERED_CSV   (default outputs/alpaca_news_tagged_filtered.csv)
    <CONTENT_COLUMNS>, symbols, created_at   (all other columns are ignored)
  config.MODEL_DIR           (default models/finbert) tokenizer, loaded offline

Tokenization
  [CLS] / [SEP] added, texts longer than 512 tokens truncated, shorter ones
  padded with [PAD] up to 512.

Output
  config.NEWS_TOKENIZED_CSV   (default outputs/alpaca_news_tagged_filtered_tokenized.csv)
    <CONTENT_COLUMNS>_tokens   512 token ids, written as a list string "[101, ...]"
    <CONTENT_COLUMNS>_mask     512 values, 1 = real token, 0 = [PAD]
    symbols                ticker, unchanged
    created_at             publish time, unchanged
"""

import pandas as pd
from transformers import AutoTokenizer

import config

CONTENT_COLUMNS = 'summary_filtered'

# Load the preprocessed data from the previous step
df = pd.read_csv(config.NEWS_FILTERED_CSV)
df = df[[CONTENT_COLUMNS, 'symbols', 'created_at']]
df = df.dropna()   # empty text is read as float NaN, which the tokenizer rejects

# Tokenize the CONTENT_COLUMNS column into a list of words, and store it in a new column called '<CONTENT_COLUMNS>_tokens'
tokenizer = AutoTokenizer.from_pretrained(str(config.MODEL_DIR), local_files_only=True)
encoded = tokenizer(
    df[CONTENT_COLUMNS].tolist(),
    add_special_tokens=True,
    truncation=True,                 # cut anything longer than max_length
    padding='max_length',            # top anything shorter up to max_length with [PAD]
    max_length=512
)
df[f'{CONTENT_COLUMNS}_tokens'] = encoded['input_ids']
df[f'{CONTENT_COLUMNS}_mask'] = encoded['attention_mask']   # 1 = real token, 0 = [PAD]; the model must not attend to padding
df = df[[f'{CONTENT_COLUMNS}_tokens', f'{CONTENT_COLUMNS}_mask', 'symbols', 'created_at']]

# Save the tokenized data to a new CSV file
config.NEWS_TOKENIZED_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(config.NEWS_TOKENIZED_CSV, index=False)