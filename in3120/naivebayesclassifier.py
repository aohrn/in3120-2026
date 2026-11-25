# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple
from .dictionary import InMemoryDictionary
from .analyzer import Analyzer
from .corpus import Corpus


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    See also https://web.stanford.edu/~jurafsky/slp3/B.pdf for supplementary material.
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
        # Maximum likelihood estimate.
        total_count = sum(map(len, training_set.values()))
        self._priors = {category: math.log(corpus.size() / total_count) for category, corpus in training_set.items()}

    def _compute_vocabulary(self, training_set, fields) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        """
        # We're doing simple add-one (Laplace) smoothing when estimating the probabilities, so
        # figure out the size of the overall vocabulary.
        for _, corpus in training_set.items():
            for document in corpus:
                for field in fields:
                    for term in self._get_terms(document.get_field(field, "")):
                        self._vocabulary.add_if_absent(term)

    def _compute_posteriors(self, training_set, fields) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        # Use smoothed estimates. Remember the denominators we used, so that we later know how
        # to deal with terms we haven't seen for a category.
        for category, corpus in training_set.items():
            term_frequencies = Counter(self._get_terms(" ".join(d.get_field(f, "") for d in corpus for f in fields)))
            self._denominators[category] = sum(term_frequencies.values()) + self._vocabulary.size()
            self._posteriors[category] = {term: self._smooth(frequency, category) for term, frequency in term_frequencies.items()}

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

    def generate(self, length: int = 100, category: str | None = None, seed: int | None = None) -> Tuple[str, str]:
        """
        Generates a new text fragment of a given length, based on the language model of the
        given category. If no category is specified, we sample a category from the prior
        distribution. An optional seed can be specified to make the generation deterministic.
        Returns a (category, text) pair.

        A naive Bayes classifier is a generative text classifier where each target class is
        modeled as a distinct unigram language model, i.e., a language model that simply throws
        away all conditioning context. We can use it to generate text that is representative of
        a given class, i.e., given the language model of a category we can generate or synthesize
        new content that is representative of that category. Since it's a unigram language model
        the generated text won't be very coherent, but it will be representative of the category's
        vocabulary and the term distribution in the generated text will follow that of the
        classifier's training set for the given category.
        """
        # For determinism in tests, if specified.
        if seed is not None:
            random.seed(seed)

        # Sample a category from the prior distribution, unless one is already specified.
        if category is None:
            categories = list(self._priors.keys())
            probabilities = [math.exp(self._priors[c]) for c in categories]
            category = random.choices(categories, weights=probabilities)[0]

        # Sample a sequence of terms from the category-conditioned posterior distribution.
        terms = list(self._posteriors[category].keys())
        probabilities = [math.exp(self._posteriors[category][t]) for t in terms]
        generated = " ".join(random.choices(terms, weights=probabilities, k=length))
        return (category, generated)

    def classify(self, buffer: str) -> Iterator[Result]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.
        """
        # Only consider terms that occurred in the training set, for any category.
        terms = [term for term in self._get_terms(buffer) if term in self._vocabulary]

        # Compute log-probabilities for each category using priors and posteriors.
        scores = {category: prior + sum(self.get_posterior(category, term) for term in terms) for category, prior in self._priors.items()}

        # Emit categories back to the client in sorted order.
        ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        yield from (self.Result(category, score) for category, score in ordered)
