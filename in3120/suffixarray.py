# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long

from bisect import bisect_left
from itertools import takewhile
from dataclasses import dataclass
from typing import Iterator, Iterable, Tuple, List
from collections import Counter
from .document import Document
from .corpus import Corpus
from .analyzer import Analyzer


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    @dataclass
    class Options:
        """
        Query-time options. Controls lookup behavior.
        """
        hit_count: int = 10  # The maximum number of results to return to the client.

    @dataclass
    class Result:
        """
        An individual lookup result, as reported back to the client.
        """
        document: Document  # The document with the matching content.
        score: int          # The number of times the query appears in the matching content.

    def __init__(self, corpus: Corpus, fields: Iterable[str], analyzer: Analyzer):
        self._corpus = corpus
        self._analyzer = analyzer
        self._haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self._suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self._build_suffix_array(fields)  # Construct the haystack and the suffix array itself.

    def _build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _get_suffix(self, pair: Tuple[int, int]) -> str:
        """
        Produces the suffix/substring from the normalized document buffer for the given (index, offset) pair.
        """
        index, offset = pair
        return self._haystack[index][1][offset:]  # Slicing implies copying. This should be possible to avoid.

    def evaluate(self, query: str, options: Options | None = None) -> Iterator[Result]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
