# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from dataclasses import dataclass
from .ranker import Ranker
from .corpus import Corpus
from .posting import Posting
from .invertedindex import InvertedIndex


class BetterRanker(Ranker):
    """
    A ranker that does traditional TF-IDF ranking, possibly combining it with
    a static document score (if present). Basically does a weighted sum of the
    two factors to produce net-score(q, d):

        net-score(q, d) = c₁ * static-score(d) + c₂ * dynamic-score(q, d)

    Here, static-score(d) is what the textbook denotes g(d), and dynamic-score(q, d)
    is given by Equation 6.9 in the textbook with an additional query multiplicity
    factor thrown in for good measure. See Section 6.2.2 and Section 7.1.4 in
    https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf for details.

    For the dynamic-score(q, d) component, it would be natural to extend the ranker
    further with, e.g., document length considerations or term frequency saturation
    effects. See, e.g., https://en.wikipedia.org/wiki/Okapi_BM25.

    If documents have an associated publication date, it would be natural to extend
    the ranker further such that freshness (i.e., the document's current age) is taken
    into account.
    """

    @dataclass
    class Options:
        """
        Configuration options for the ranker.
        """
        static_field: str = "static_quality_score"  # The named document field that holds g(d).
        static_default: float = 0.0                 # The default g(d) value, if absent.
        static_weight: float = 1.0                  # The constant c₁ above.
        dynamic_weight: float = 1.0                 # The constant c₂ above.

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex, options: Options | None = None):
        self._score = 0.0
        self._document_id: int | None = None
        self._corpus = corpus
        self._inverted_index = inverted_index
        self._options = options or self.Options()

    def reset(self, document_id: int) -> None:
        self._score = 0.0
        self._document_id = document_id

    def update(self, term: str, multiplicity: int, posting: Posting) -> None:
        assert term is not None
        assert multiplicity > 0
        assert posting is not None
        assert posting.term_frequency > 0
        assert posting.document_id == self._document_id
        tf_score = 1.0 + math.log10(posting.term_frequency)
        idf_score = math.log10(self._corpus.size() / self._inverted_index.get_document_frequency(term))
        self._score += (1.0 + math.log10(multiplicity)) * tf_score * idf_score

    def evaluate(self) -> float:
        # Now that the dynamic (query-dependent) score is fully updated, combine it
        # with the static (query-independent) score using a simple weighted sum. Other
        # ways of combining the two are plausible. In a large real-world search system,
        # weights would be machine-learnt offline and it'd be up to the chosen ML model
        # how to best combine the many features to yield a compound relevance score.
        assert self._document_id is not None
        document = self._corpus[self._document_id]
        static_quality_score = float(document[self._options.static_field] or self._options.static_default)
        return (self._options.dynamic_weight * self._score) + (self._options.static_weight * static_quality_score)
