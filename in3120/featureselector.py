#pylint: disable=missing-module-docstring
#pylint: disable=missing-function-docstring
#pylint: disable=line-too-long

from typing import Iterable, Iterator, Dict, Callable
from dataclasses import dataclass
from collections import Counter, defaultdict
from math import log2
from .sieve import Sieve
from .analyzer import Analyzer
from .corpus import Corpus
from .invertedindex import DummyInMemoryInvertedIndex
from .trie import Trie


class FeatureSelector:
    """
    Identifies the strongest features for each category in a training set, so that these
    can subsequently be used by the client to build a trimmed-down classifier.

    Feature selection serves two main purposes: First, it makes training and applying a
    classifier more efficient by decreasing the size of the effective vocabulary. This is
    of particular importance for classifiers that are expensive to train. Second, feature
    selection often increases classification accuracy by eliminating noise features. A noise
    feature is one that, when added to the document representation, increases the classification
    error on new data. See Section 13.5 in https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
    for details.

    As currently implemented, neither χ² nor mutual information distinguish between positively
    and negatively correlated features. Because most good text classification features are
    positively correlated (i.e., they occur more often in c than in not-c), one may want to
    explicitly rule out the selection of negative indicators.
    """

    @dataclass
    class Result:
        """
        An individual scored feature, as reported back to the client.
        """
        category: str  # The category in question.
        term: str      # The term/feature in question.
        score: float   # The utility score as assessed by the feature selector.

    @dataclass
    class ContingencyTable:
        """
        Internal helper class. Represents a 2x2 contingency table, with convenience methods
        for computing row and column sums. Use the notation introduced in Section 13.5.1 in
        the textbook.
        """
        n_00: int = 0  # Number of documents that do not contain term t and are not in category c.
        n_01: int = 0  # Number of documents that do not contain term t and are in category c.
        n_10: int = 0  # Number of documents that contain term t and are not in category c.
        n_11: int = 0  # Number of documents that contain term t and are in category c.
        @property
        def n_0x(self) -> int:
            return self.n_00 + self.n_01
        @property
        def n_x0(self) -> int:
            return self.n_00 + self.n_10
        @property
        def n_1x(self) -> int:
            return self.n_10 + self.n_11
        @property
        def n_x1(self) -> int:
            return self.n_01 + self.n_11
        @property
        def n_xx(self) -> int:
            return self.n_00 + self.n_01 + self.n_10 + self.n_11

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str], analyzer: Analyzer):
        self._sizes = {category: corpus.size() for category, corpus in training_set.items()}
        self._counters = self._compute_counts(training_set, fields, analyzer)

    def _compute_counts(self, training_set: Dict[str, Corpus], fields: Iterable[str], analyzer: Analyzer) -> Dict[str, Counter]:
        """
        Computes the document counts for each term for each category in the training set.
        """
        counters: Dict[str, Counter] = defaultdict(Counter)
        for category, corpus in training_set.items():
            dummy = DummyInMemoryInvertedIndex(corpus, fields, analyzer)
            for term in dummy.get_indexed_terms():
                counters[category][term] += dummy.get_document_frequency(term)
        return counters

    def _mutual_information(self, table: ContingencyTable) -> float:
        """
        Computes the expected mutual information metric I(U; C) between a category and a term,        
        given values in the contingency table computed from a (category, term) pair and using
        maximum likelihood estimates for the probabilities.
        """
        # See Equation 13.17 in https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf for details.
        score = 0.0
        score += (table.n_11 / table.n_xx) * log2((table.n_xx * table.n_11) / (table.n_1x * table.n_x1)) if table.n_11 > 0 else 0.0
        score += (table.n_01 / table.n_xx) * log2((table.n_xx * table.n_01) / (table.n_0x * table.n_x1)) if table.n_01 > 0 else 0.0
        score += (table.n_10 / table.n_xx) * log2((table.n_xx * table.n_10) / (table.n_1x * table.n_x0)) if table.n_10 > 0 else 0.0
        score += (table.n_00 / table.n_xx) * log2((table.n_xx * table.n_00) / (table.n_0x * table.n_x0)) if table.n_00 > 0 else 0.0
        return score

    def _chi_square(self, table: ContingencyTable) -> float:
        """
        Computes the X² metric between a category and a term, given values in the contingency
        table computed from a (category, term) pair.
        """
        # See Equation 13.19 in https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf for details.
        return (table.n_xx * ((table.n_11 * table.n_00) - (table.n_10 * table.n_01))**2) / (table.n_x1 * table.n_1x * table.n_x0 * table.n_0x)

    def _select_features(self, categories: Iterable[str], k: int, stopwords: Trie | None, utility: Callable[[ContingencyTable], float]) -> Iterator[Result]:
        """
        For each given category, yields the category's top k features according to the given utility function.
        A dictionary with known features to ignore in the scoring process can optionally be supplied.

        See Figure 13.6 in https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf for details.
        """
        assert k > 0
        assert all(category in self._counters for category in categories), "Unknown category."

        for category in categories:
            sieve = Sieve(k)
            for term in self._counters[category]:
                if stopwords is not None and term in stopwords:
                    continue
                table = self.ContingencyTable()
                table.n_10 = sum(self._counters[c][term] for c in self._counters if c != category)
                table.n_11 = self._counters[category][term]
                table.n_01 = self._sizes[category] - table.n_11
                table.n_00 = sum(self._sizes[c] for c in self._counters if c != category) - table.n_10
                sieve.sift(utility(table), term)
            yield from (self.Result(category, term, score) for score, term in sieve.winners())

    def mutual_information(self, categories: Iterable[str], k: int, stopwords: Trie | None) -> Iterator[Result]:
        """
        Does feature selection according to the mutual information criterion, and yields the
        highest-scoring terms for each category back to the client.

        Mutual information I(U; C) measures how much information (in the information-theoretic sense)
        a term contains about the category. If a term's distribution is the same in the category
        as it is in the collection as a whole, then I(U; C) = 0. If the term is a perfect indicator
        for category membership, then I(U; C) = 1. Mutual information reaches its maximum value if
        the term is a perfect indicator for category membership, that is, if the term is present in
        a document if and only if the document belongs to the category.

        See Section 13.5.1 in https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf for details.
        """
        return self._select_features(categories, k, stopwords, self._mutual_information)

    def chi_square(self, categories: Iterable[str], k: int, stopwords: Trie | None) -> Iterator[Result]:
        """
        Does feature selection according to the χ² criterion, and yields the highest-scoring terms
        for each category back to the client.

        See Section 13.5.2 in https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf for details.
        """
        return self._select_features(categories, k, stopwords, self._chi_square)
