"""Filter the tagged Alpaca news and clean its text.

Keeps only the articles with tag == 1 that mention exactly one ticker, then
strips headline, summary and content down to plain ASCII prose.

Input
  config.NEWS_TAGGED_CSV   (default data/alpaca_news_tagged.csv)
    id, headline, summary, content, symbols, created_at, tag
    - tag:     1 = keep, 0 = drop
    - symbols: comma-separated tickers; rows with more than one are dropped

Cleaning (same chain on headline, summary and content)
  &amp; -> &, other HTML entities / <tags> / CSS rule blocks / URLs / e-mails -> space,
  curly quotes, dashes and ellipsis -> ASCII, NFKD + drop all remaining non-ASCII,
  any character outside [A-Za-z0-9 .,!?;:'"()%$&/+-] -> space, collapse whitespace.
  Missing values become ''.

Output
  config.NEWS_FILTERED_CSV   (default outputs/alpaca_news_tagged_filtered.csv)
    id, headline, summary, content, symbols, created_at   (original columns, tag removed)
    body                raw headline + ' ' + content (NaN if either is missing)
    headline_filtered   cleaned headline
    summary_filtered    cleaned summary
    content_filtered    cleaned content
    body_filtered       headline_filtered + ' ' + content_filtered
"""

import pandas as pd

import config

# Drop rows with tag 0 and rows with multiple symbols
df = pd.read_csv(config.NEWS_TAGGED_CSV)
df = df[df['tag'] == 1].drop(columns=['tag'])
df = df[~df['symbols'].str.contains(',')]

# Merge the headline and content columns into a single column called 'body'
df['body'] = df['headline'] + ' ' + df['content']

# Clean the 'headline' column by removing HTML tags, CSS rules, URLs, and unwanted characters
headline = df['headline'].fillna('')                                                      # NaN headline (no content) -> '' so .str works
headline = headline.str.replace('&amp;', '&', regex=False)                                # the one HTML entity in this feed: &amp; -> &
headline = headline.str.replace(r'&[a-zA-Z#][a-zA-Z0-9]{1,8};', ' ', regex=True)          # any other stray entity, e.g. &nbsp; &#8217;
headline = headline.str.replace(r'<[^>]{1,200}>', ' ', regex=True)                        # leftover <...> markup
headline = headline.str.replace(r'[^\s{}]+(?:\s*[,>+~]\s*[^\s{}]+)*\s*\{[^{}]*\}', ' ', regex=True)  # stylesheet blocks: th, td { padding: 8px; }
headline = headline.str.replace(r'(?:https?://|www\.)\S+|\S+@\S+\.\S+', ' ', regex=True)  # links and e-mail addresses
headline = headline.str.replace(r"[‘’‚‛]", "'", regex=True)                               # curly single quotes -> ASCII apostrophe
headline = headline.str.replace(r'[“”„]', '"', regex=True)                                # curly double quotes -> ASCII quote
headline = headline.str.replace(r'[‐‑–—−]', '-', regex=True)                              # hyphen, en/em dash, minus -> ASCII hyphen
headline = headline.str.replace('…', '...', regex=False)                                  # one-character ellipsis -> three dots
headline = headline.str.replace(r'[•·]', ' ', regex=True)                                 # bullets -> space, so neighbours stay apart
headline = headline.str.normalize('NFKD')                                                 # NBSP -> space, accents split off the letter
headline = headline.str.encode('ascii', 'ignore').str.decode('ascii')                     # every non-ASCII left: emoji, CJK, Greek, currency
headline = headline.str.replace(r"""[^A-Za-z0-9 .,!?;:'"()%$&/+-]""", ' ', regex=True)    # ASCII specials: @ # * [ ] { } ~ = _ < >
df['headline_filtered'] = headline.str.replace(r'\s+', ' ', regex=True).str.strip()                # collapse the gaps left behind, trim ends

# Clean the 'summary' column by removing HTML tags, CSS rules, URLs, and unwanted characters
summary = df['summary'].fillna('')                                                      # NaN summary (no content) -> '' so .str works
summary = summary.str.replace('&amp;', '&', regex=False)                                # the one HTML entity in this feed: &amp; -> &
summary = summary.str.replace(r'&[a-zA-Z#][a-zA-Z0-9]{1,8};', ' ', regex=True)          # any other stray entity, e.g. &nbsp; &#8217;
summary = summary.str.replace(r'<[^>]{1,200}>', ' ', regex=True)                        # leftover <...> markup
summary = summary.str.replace(r'[^\s{}]+(?:\s*[,>+~]\s*[^\s{}]+)*\s*\{[^{}]*\}', ' ', regex=True)  # stylesheet blocks: th, td { padding: 8px; }
summary = summary.str.replace(r'(?:https?://|www\.)\S+|\S+@\S+\.\S+', ' ', regex=True)  # links and e-mail addresses
summary = summary.str.replace(r"[‘’‚‛]", "'", regex=True)                               # curly single quotes -> ASCII apostrophe
summary = summary.str.replace(r'[“”„]', '"', regex=True)                                # curly double quotes -> ASCII quote
summary = summary.str.replace(r'[‐‑–—−]', '-', regex=True)                              # hyphen, en/em dash, minus -> ASCII hyphen
summary = summary.str.replace('…', '...', regex=False)                                  # one-character ellipsis -> three dots
summary = summary.str.replace(r'[•·]', ' ', regex=True)                                 # bullets -> space, so neighbours stay apart
summary = summary.str.normalize('NFKD')                                                 # NBSP -> space, accents split off the letter
summary = summary.str.encode('ascii', 'ignore').str.decode('ascii')                     # every non-ASCII left: emoji, CJK, Greek, currency
summary = summary.str.replace(r"""[^A-Za-z0-9 .,!?;:'"()%$&/+-]""", ' ', regex=True)    # ASCII specials: @ # * [ ] { } ~ = _ < >
df['summary_filtered'] = summary.str.replace(r'\s+', ' ', regex=True).str.strip()                # collapse the gaps left behind, trim ends

# Clean the 'content' column by removing HTML tags, CSS rules, URLs, and unwanted characters
content = df['content'].fillna('')                                                      # NaN content (no content) -> '' so .str works
content = content.str.replace('&amp;', '&', regex=False)
content = content.str.replace(r'&[a-zA-Z#][a-zA-Z0-9]{1,8};', ' ', regex=True)          # any other stray entity, e.g. &nbsp; &#8217;
content = content.str.replace(r'<[^>]{1,200}>', ' ', regex=True)                        # leftover <...> markup
content = content.str.replace(r'[^\s{}]+(?:\s*[,>+~]\s*[^\s{}]+)*\s*\{[^{}]*\}', ' ', regex=True)  # stylesheet blocks: th, td { padding: 8px; }
content = content.str.replace(r'(?:https?://|www\.)\S+|\S+@\S+\.\S+', ' ', regex=True)  # links and e-mail addresses
content = content.str.replace(r"[‘’‚‛]", "'", regex=True)                               # curly single quotes -> ASCII apostrophe
content = content.str.replace(r'[“”„]', '"', regex=True)                                # curly double quotes -> ASCII quote
content = content.str.replace(r'[‐‑–—−]', '-', regex=True)                              # hyphen, en/em dash, minus -> ASCII hyphen
content = content.str.replace('…', '...', regex=False)                                  # one-character ellipsis -> three dots
content = content.str.replace(r'[•·]', ' ', regex=True)                                 # bullets -> space, so neighbours stay apart
content = content.str.normalize('NFKD')                                                 # NBSP -> space, accents split off the letter
content = content.str.encode('ascii', 'ignore').str.decode('ascii')                     # every non-ASCII left: emoji, CJK, Greek, currency
content = content.str.replace(r"""[^A-Za-z0-9 .,!?;:'"()%$&/+-]""", ' ', regex=True)    # ASCII specials: @ # * [ ] { } ~ = _ < >
df['content_filtered'] = content.str.replace(r'\s+', ' ', regex=True).str.strip()                # collapse the gaps left behind, trim ends

# Combine the filtered headline and content into a single column called 'body_filtered'
df['body_filtered'] = df['headline_filtered'] + ' ' + df['content_filtered']

# Clean the 'body' column by removing HTML tags, CSS rules, URLs, and unwanted characters
# body = df['body'].fillna('')                                                      # NaN body (no content) -> '' so .str works
# body = body.str.replace('&amp;', '&', regex=False)                                # the one HTML entity in this feed: &amp; -> &
# body = body.str.replace(r'&[a-zA-Z#][a-zA-Z0-9]{1,8};', ' ', regex=True)          # any other stray entity, e.g. &nbsp; &#8217;
# body = body.str.replace(r'<[^>]{1,200}>', ' ', regex=True)                        # leftover <...> markup
# body = body.str.replace(r'[^\s{}]+(?:\s*[,>+~]\s*[^\s{}]+)*\s*\{[^{}]*\}', ' ', regex=True)  # stylesheet blocks: th, td { padding: 8px; }
# body = body.str.replace(r'(?:https?://|www\.)\S+|\S+@\S+\.\S+', ' ', regex=True)  # links and e-mail addresses
# body = body.str.replace(r"[‘’‚‛]", "'", regex=True)                               # curly single quotes -> ASCII apostrophe
# body = body.str.replace(r'[“”„]', '"', regex=True)                                # curly double quotes -> ASCII quote
# body = body.str.replace(r'[‐‑–—−]', '-', regex=True)                              # hyphen, en/em dash, minus -> ASCII hyphen
# body = body.str.replace('…', '...', regex=False)                                  # one-character ellipsis -> three dots
# body = body.str.replace(r'[•·]', ' ', regex=True)                                 # bullets -> space, so neighbours stay apart
# body = body.str.normalize('NFKD')                                                 # NBSP -> space, accents split off the letter
# body = body.str.encode('ascii', 'ignore').str.decode('ascii')                     # every non-ASCII left: emoji, CJK, Greek, currency
# body = body.str.replace(r"""[^A-Za-z0-9 .,!?;:'"()%$&/+-]""", ' ', regex=True)    # ASCII specials: @ # * [ ] { } ~ = _ < >
# df['body_filtered'] = body.str.replace(r'\s+', ' ', regex=True).str.strip()                # collapse the gaps left behind, trim ends

# Save the filtered dataframe to a new CSV file
config.NEWS_FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(config.NEWS_FILTERED_CSV, index=False)
