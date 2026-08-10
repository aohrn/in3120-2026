# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=line-too-long

from abc import ABC
from typing import Iterator, Tuple
from .normalizer import Normalizer, SimpleNormalizer, DummyNormalizer
from .tokenizer import Span, Tokenizer, SimpleTokenizer, DummyTokenizer


class Analyzer(ABC):
    """
    Combines a normalizer and a tokenizer in order to produce a stream of
    normalized terms. Deals with buffer canonicalization, tokenization,
    and normalization of tokens. Terms can be dropped, too. Note that normalizers
    can be chained together, and you can pass in the chain as a single normalizer.

    Open-source systems based on Lucene have an almost identical concept, e.g.:

    - Lucene (https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/analysis/Analyzer.html)
    - Elasticsearch (https://www.elastic.co/guide/en/elasticsearch/reference/current/analyzer-anatomy.html)
    - Solr (https://solr.apache.org/guide/solr/latest/indexing-guide/analyzers.html)
    """

    def __init__(self, normalizer: Normalizer, tokenizer: Tokenizer):
        self._normalizer = normalizer
        self._tokenizer = tokenizer

    @property
    def normalizer(self) -> Normalizer:
        """
        Returns the analyzer's normalizer.
        """
        return self._normalizer

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Returns the analyzer's tokenizer.
        """
        return self._tokenizer

    def join(self, buffer: str, canonicalize: bool = True) -> str:
        """
        Returns a processed version of the given buffer, where the normalized terms
        have been joined back together into a single buffer.
        """
        return self._tokenizer.join(self.terms(buffer, canonicalize))

    def spans(self, buffer: str, canonicalize: bool = True) -> Iterator[Span]:
        """
        Returns the positional spans (ranges) that indicate where in the given buffer
        the terms begin and end.
        """
        buffer = self._normalizer.canonicalize(buffer) if canonicalize else buffer
        return self._tokenizer.spans(buffer)

    def terms(self, buffer: str, canonicalize: bool = True) -> Iterator[Tuple[str, Span]]:
        """
        Returns the (term, span) pairs that appear in the given buffer. Tokens that
        normalize to None or the empty string get dropped.
        """
        buffer = self._normalizer.canonicalize(buffer) if canonicalize else buffer
        tokens = self._tokenizer.tokens(buffer)
        return ((term, span) for string, span in tokens if (term := self._normalizer.normalize(string)))


class SimpleAnalyzer(Analyzer):
    """
    An analyzer that uses simple defaults.
    """

    def __init__(self):
        super().__init__(SimpleNormalizer(), SimpleTokenizer())


class DummyAnalyzer(Analyzer):
    """
    An analyzer that doesn't really do anything.
    """

    def __init__(self):
        super().__init__(DummyNormalizer(), DummyTokenizer())
