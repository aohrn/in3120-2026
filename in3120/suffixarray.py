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
        # We allow searching across multiple document fields simultaneously, so join the named fields
        # to produce the haystack that we'll search for needles in. Avoid cross-field matches.
        self._haystack = [(d.document_id, " \0 ".join(self._analyzer.join(d.get_field(f, "")) for f in fields)) for d in self._corpus]

        # We don't actually store all suffixes, instead we store (index, offset) pairs which allows us
        # to generate the suffixes if/when we need them: The index identifies the document, and the
        # offset identifies where in the document the substring starts. A naive suffix array generation
        # is fine for now.
        self._suffixes = [(index, begin) for index, (_, buffer) in enumerate(self._haystack) for begin, _ in self._analyzer.spans(buffer, False)]
        self._suffixes.sort(key=self._get_suffix)

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
        # Default options apply unless specified.
        options = options or self.Options()

        # Search for the needle in the haystack, using built-in binary search. Define that the empty query matches
        # nothing, not everything.
        needle = self._analyzer.join(query or "")
        if not needle:
            return
        where_start = bisect_left(self._suffixes, needle, key=self._get_suffix)

        # Helper predicate. Checks if the identified suffix starts with the needle. Since slicing implies copying,
        # cap the length of the slice to the length of the needle. The starts-with relation then becomes the same
        # as equality, which is quick to check.
        def _is_match(i: int) -> bool:
            j, offset = self._suffixes[i]
            return self._haystack[j][1][offset:(offset + len(needle))] == needle

        # Suffixes sharing a prefix are consecutive in the suffix array. Scan ahead from the located index until
        # we no longer get a match. We expect a low number of matches for typical queries, and we process all the
        # matches below anyway. If we just wanted to count the number of matches without processing them, we
        # could instead of a linear scan do another binary search to locate where the range ends.
        matches = takewhile(_is_match, range(where_start, len(self._suffixes)))

        # Deduplicate. A document in the haystack might contain multiple occurrences of the needle.
        # Rank according to occurrence count, and emit in ranked order.
        if matches:
            pairs = (self._suffixes[i] for i in matches)
            winners = Counter(i for i, _ in pairs).most_common(max(1, min(100, options.hit_count)))
            yield from (self.Result(self._corpus[self._haystack[index][0]], count) for index, count in winners)
