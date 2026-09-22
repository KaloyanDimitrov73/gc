from typing import List, cast

import bm25s

_TOKEN_PATTERN = r"[a-z0-9]+"


def tokenize_for_bm25(texts: List[str]) -> List[List[str]]:
    """Tokenizes texts consistently for BM25 indexing and querying."""
    return cast(
        List[List[str]],
        bm25s.tokenize(
            texts,
            lower=True,
            token_pattern=_TOKEN_PATTERN,
            stopwords=None,
            return_ids=False,
            show_progress=False,
        ),
    )
