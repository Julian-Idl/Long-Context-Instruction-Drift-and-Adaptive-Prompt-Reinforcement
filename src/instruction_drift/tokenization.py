"""Reference tokenizer.

Conversation design (positions, distances, densities) is measured with one fixed
reference tokenizer so the same conversation plan is comparable across target models.
Provider-reported token usage is logged separately and never mixed with these counts.
Counts cover message content only, not provider chat-template overhead.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

REFERENCE_ENCODING = "o200k_base"


class Tokenizer:
    def __init__(self, encoding: str = REFERENCE_ENCODING):
        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}:{tiktoken.__version__}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self._enc.encode(text, disallowed_special=())
        return text if len(ids) <= max_tokens else self._enc.decode(ids[:max_tokens])


@lru_cache(maxsize=1)
def reference_tokenizer() -> Tokenizer:
    return Tokenizer()
