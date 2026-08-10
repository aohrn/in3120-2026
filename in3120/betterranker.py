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
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def update(self, term: str, multiplicity: int, posting: Posting) -> None:
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def evaluate(self) -> float:
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
