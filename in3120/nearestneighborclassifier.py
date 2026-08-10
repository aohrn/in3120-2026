# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Set
from collections import defaultdict
from .corpus import Corpus, InMemoryCorpus
from .analyzer import Analyzer
from .similaritysearchengine import SimilaritySearchEngine


class NearestNeighborClassifier:
    """
    Implements a simple k-NN text classifier. Compared to the reference functionality as described
    in https://nlp.stanford.edu/IR-book/html/htmledition/k-nearest-neighbor-1.html, the implementation
    is slightly generalized to support the case where an example in the training set can belong to
    multiple categories, and where we can do both simple voting and similarity-based voting. 
    """

    @dataclass
    class Options:
        """
        Query-time options. Controls lookup behavior.
        """
        k: int = 3              # The number of nearest neighbors to consider.
        voting: str = "simple"  # How the nearest neighbors cast their votes.

    @dataclass
    class Result:
        """
        An individual classification result, as reported back to the client.
        """
        category: str  # The category that the classifier maps the buffer into.
        score: float   # Indicates how good a fit the classifier finds the category to be for the buffer.

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str], analyzer: Analyzer):
        """
        Creates a k-NN text classifier. Indexes all the documents in the training set
        so that we can do efficient text classification later on.
        """
        # Create a mapping from document identifiers to their categories. A document
        # can belong to more than one category.
        self._categories: Dict[int, Set[str]] = defaultdict(set)
        for category, corpus in training_set.items():
            for document in corpus:
                self._categories[document.document_id].add(category)

        # Create a merged corpus of all the unique documents, which is what we'll index.
        merged = InMemoryCorpus.merge(training_set)

        # This does the heavy lifting and retrieves the most similar training examples.
        self._retriever = SimilaritySearchEngine(merged, fields, analyzer)

        # Alternative voting strategies.
        self._voters = {
            "simple":   lambda _: 1,
            "weighted": lambda m: m.score,
        }

    def classify(self, buffer: str, options: Options | None = None) -> Iterator[Result]:
        """
        Classifies the given buffer according to the k-NN classification algorithm.
        """
        # Default options apply unless specified.
        options = options or self.Options()

        # Retrieve the most similar documents in the set of training examples. The cosine
        # similarity to each matching training example is also returned.
        matches = list(self._retriever.evaluate(buffer, SimilaritySearchEngine.Options(hit_count=options.k)))

        # How should we tally up the votes? We either do simple voting, or we weight
        # each vote by the similarity to the training example.
        assert options.voting in self._voters
        voter = self._voters[options.voting]

        # Score the categories that the matching documents belong to.
        totals: Dict[str, float] = defaultdict(float)
        accumulated = sum(voter(match) for match in matches)

        for match in matches:
            for category in self._categories[match.document.document_id]:
                totals[category] += voter(match)

        # Yield results back to the client in sorted order. Normalize the scores.
        ordered = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        yield from (self.Result(category, score / accumulated) for category, score in ordered)
