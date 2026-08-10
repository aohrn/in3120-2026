# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator
from .dictionary import InMemoryDictionary
from .analyzer import Analyzer
from .corpus import Corpus


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    """

    @dataclass
    class Result:
        """
        An individual classification result, as reported back to the client.
        """
        category: str  # The category that the classifier maps the buffer into.
        score: float   # The log-probability as assessed by the classifier.

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str], analyzer: Analyzer):
        # Used for breaking the text up into discrete classification features.
        self._analyzer = analyzer

        # The vocabulary we've seen during training.
        self._vocabulary = InMemoryDictionary()

        # Maps a category c to the logarithm of its prior probability, i.e., c maps to log(Pr(c)).
        self._priors: Dict[str, float] = {}

        # Maps a category c and a term t to the logarithm of its category-conditioned posterior probability,
        # i.e., (c, t) maps to log(Pr(t | c)).
        self._posteriors: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing for the posterior probabilities.
        self._denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self._compute_priors(training_set)
        self._compute_vocabulary(training_set, fields)
        self._compute_posteriors(training_set, fields)

    def _compute_priors(self, training_set) -> None:
        """
        Estimates all prior probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _compute_vocabulary(self, training_set, fields) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _compute_posteriors(self, training_set, fields) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _smooth(self, frequency: int, category: str) -> float:
        """
        Computes a smoothed log-probability, using Lapace add-one smoothing. Assumes that
        we've already computed the correct fraction denominator to use for the given category.
        """
        return math.log((frequency + 1) / self._denominators[category])

    def _get_terms(self, buffer) -> Iterator[str]:
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        return (t for t, _ in self._analyzer.terms(buffer))

    def get_prior(self, category: str) -> float:
        """
        Given a category c, returns the category's prior log-probability log(Pr(c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        return self._priors[category]

    def get_posterior(self, category: str, term: str) -> float:
        """
        Given a category c and a term t, returns the posterior log-probability log(Pr(t | c)).
        If the term has not been observed for the current category, use a smoothed estimate.

        This is an internal detail having public visibility to facilitate testing.
        """
        return self._posteriors[category].get(term, self._smooth(0, category))

    def classify(self, buffer: str) -> Iterator[Result]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
