# Sparse index stores

This package contains the persistent stores used by HubLink's two sparse
retrieval channels: BM25 and SPLADE. Both channels index the same logical
documents, but they represent and persist those documents differently.

## Indexed document

Sparse indexing treats each `HubPath` as one searchable document. The document
is created from the path as follows:

```python
SparseIndexDocument(
    record_id=f"{hub_id}:{path.path_hash}",
    text=path.path_text,
    hub_id=hub_id,
    path_hash=path.path_hash,
)
```

Consequently:

- one index row represents one hub path, not an entire hub;
- `path.path_text` is the text indexed by both channels;
- `hub_id` and `path_hash` map a search result back to its logical `HubPath`;
- the triples in `path.path` are not separately indexed unless their content is
  already included in `path.path_text`.

The same sparse-index key is used to identify the BM25 and SPLADE artifacts for
a configuration, but the artifacts themselves are separate.

## BM25

BM25 indexes a lexical representation of the complete `path.path_text`. It
does not store a manually selected list of keywords.

### Indexing

Each path text is lowercased and tokenized using the pattern `[a-z0-9]+`.
Tokenization does not reduce the document to a unique keyword list: token
repetitions and the complete document length are retained as inputs to BM25.

```text
path_0: "graph database"
        -> ["graph", "database"]
```

No stop-word list is applied. From the complete token sequences, BM25S derives
statistics such as:

| Term | `TF(term, path_0)` | `TF(term, path_1)` | `DF(term)` |
| --- | ---: | ---: | ---: |
| `graph` | 2 | 1 | 2 |
| `database` | 0 | 1 | 1 |

Here:

- **TF**, or term frequency, is the number of occurrences of a term in one
  path text. Repeating `graph` therefore affects the path's score.
- **DF**, or document frequency, is the number of path texts containing the
  term. It is used to calculate inverse document frequency (IDF), so a rare
  term generally carries more weight than a common term.
- **Document length** is the number of tokens in a path text. In the example,
  the lengths are 3 and 2 and the average document length is 2.5. BM25 uses
  these values to prevent long paths from winning merely because they contain
  more words.

The code constructs `bm25s.BM25(method="lucene")`. For every term `t` occurring
in document `d`, its BM25 contribution is based on all of the above values:

```text
term contribution(t, d)
    = IDF(t)
      * BM25_TF_normalization(
            TF(t, d),
            length(d),
            average_document_length,
            k1,
            b
        )
```

The length-normalized TF component is proportional to:

```text
                          TF(t, d)
----------------------------------------------------------------
TF(t, d) + k1 * (1 - b + b * length(d) / average_document_length)
```

Therefore BM25 does not store only `{"graph", "retrieval", "database"}`. It
uses the complete term counts, corpus frequencies, and path lengths to compute
a different weight for each non-zero `(term, path)` pair.


### Persistence

BM25 is saved in BM25S's native directory format under:

```text
<cache>/hublink_bm25_index/<index_key>/path/
```

The native index is saved with an aligned corpus containing each row's
`record_id`, `hub_id`, and `path_hash`. The original `path.path_text` is not
included in this metadata corpus; its searchable lexical representation is in
the BM25 index.

### Search

The query is tokenized with exactly the same tokenizer, and BM25S calculates
one raw score for every indexed path:

```python
query_tokens = tokenize_for_bm25([query_text])[0]
scores = bm25_index.get_scores(query_tokens)
```

Only positive-score paths become search hits.

## SPLADE

SPLADE encodes the complete `path.path_text` into a learned sparse vector over
the SPLADE model's vocabulary.

### Indexing

Conceptually, a path becomes a weighted token map:

```text
"Marie Curie received the Nobel Prize in Physics"
                         |
                         v
{
    "marie": 1.8,
    "curie": 2.3,
    "nobel": 1.9,
    "physics": 1.4,
    "scientist": 0.7,
    ...
}
```

Unlike BM25 tokenization, SPLADE can assign weight to learned expansion terms
that are not literally present in the path text. Most vocabulary dimensions
remain zero, so the result is sparse despite having the model vocabulary as
its full dimensional space.

Each path vector is one row in a sparse PyTorch tensor:

```text
                    SPLADE model vocabulary
                token_1  token_2  ...  token_V
path_1             0.0      1.7   ...    0.0
path_2             2.1      0.0   ...    0.4
...                ...      ...   ...    ...
path_N             0.0      0.0   ...    1.3
```

The matrix shape is therefore conceptually:

```text
[number_of_indexed_hub_paths, model_vocabulary_size]
```

### Persistence

The SPLADE artifact is stored as a trusted pickle file under:

```text
<cache>/hublink_splade_index/<index_key>_path.pkl
```

Its payload contains:

```python
{
    "records": [
        {"record_id": "...", "hub_id": "...", "path_hash": "..."},
        ...,
    ],
    "hub_ids": [...],       # retained compatibility field
    "path_hashes": [...],   # retained compatibility field
    "matrix": sparse_tensor,
    "model_name": "...",
}
```

The raw path text is not persisted in this payload. Its learned representation
is stored in `matrix`, while `records` maps matrix rows back to hub paths.
