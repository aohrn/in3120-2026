# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from typing import List
from context import in3120


class TestBetterRanker(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"title": "the foo", "static_quality_score": 0.9}))
        corpus.add_document(in3120.InMemoryDocument(1, {"title": "the foo", "static_quality_score": 0.2}))
        corpus.add_document(in3120.InMemoryDocument(2, {"title": "the foo foo", "static_quality_score": 0.2}))
        corpus.add_document(in3120.InMemoryDocument(3, {"title": "the bar"}))
        corpus.add_document(in3120.InMemoryDocument(4, {"title": "the bar bar"}))
        corpus.add_document(in3120.InMemoryDocument(5, {"title": "the baz"}))
        corpus.add_document(in3120.InMemoryDocument(6, {"title": "the baz"}))
        corpus.add_document(in3120.InMemoryDocument(7, {"title": "the baz baz"}))
        index = in3120.InMemoryInvertedIndex(corpus, ["title"], analyzer)
        self._ranker = in3120.BetterRanker(corpus, index)
        options = in3120.BetterRanker.Options(dynamic_weight=0.0)
        self._custom = in3120.BetterRanker(corpus, index, options)

    def test_term_frequency(self):
        self._ranker.reset(1)
        self._ranker.update("foo", 1, in3120.Posting(1, 1))
        score1 = self._ranker.evaluate()
        self._ranker.reset(2)
        self._ranker.update("foo", 1, in3120.Posting(2, 2))
        score2 = self._ranker.evaluate()
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)
        self.assertGreater(score2, score1)

    def test_term_frequency_gets_ignored_with_custom_options(self):
        self._custom.reset(0)
        self._custom.update("foo", 1, in3120.Posting(0, 1))
        score0 = self._custom.evaluate()
        self._custom.reset(1)
        self._custom.update("foo", 1, in3120.Posting(1, 1))
        score1 = self._custom.evaluate()
        self.assertEqual(score0, 0.9)
        self.assertEqual(score1, 0.2)

    def test_document_id_mismatch(self):
        for document_id in (4, 21):
            with self.assertRaises(AssertionError):
                self._ranker.reset(document_id)
                self._ranker.update("foo", 1, in3120.Posting(42, 4))

    def test_inverse_document_frequency(self):
        self._ranker.reset(3)
        self._ranker.update("the", 1, in3120.Posting(3, 1))
        self.assertAlmostEqual(self._ranker.evaluate(), 0.0, 8)
        self._ranker.reset(3)
        self._ranker.update("bar", 1, in3120.Posting(3, 1))
        score1 = self._ranker.evaluate()
        self._ranker.reset(5)
        self._ranker.update("baz", 1, in3120.Posting(5, 1))
        score2 = self._ranker.evaluate()
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)
        self.assertGreater(score1, score2)

    def test_static_quality_score(self):
        self._ranker.reset(0)
        self._ranker.update("foo", 1, in3120.Posting(0, 1))
        score1 = self._ranker.evaluate()
        self._ranker.reset(1)
        self._ranker.update("foo", 1, in3120.Posting(1, 1))
        score2 = self._ranker.evaluate()
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)
        self.assertGreater(score1, score2)

    def test_repeated_updates_with_omnipresent_term_does_not_accumulate(self):
        term = "the"  # Appears in every document.
        scores = []
        for i in range(10):
            self._ranker.reset(0)
            for _ in range(i):
                self._ranker.update(term, 1, in3120.Posting(0, 1))
            scores.append(self._ranker.evaluate())
        self.assertTrue(all(score == scores[0] for score in scores))

    def _evaluate_cranfield(self, cranfield: in3120.TREC, engine: in3120.SimpleSearchEngine, ranker: in3120.Ranker) -> float:
        options = in3120.SimpleSearchEngine.Options(match_threshold=0.5, hit_count=10)
        score = 0.0
        for query in cranfield.queries:
            utterance = query["title"]  # This is a longish NL query that could be transformed somehow. For now, just use it as-is.
            gains = [1.0 if cranfield.is_relevant(query, result.document) else 0.0 for result in engine.evaluate(utterance, ranker, options)]
            perfect = [1.0] * min(options.hit_count, cranfield.relevant_count(query))
            score += in3120.EvaluationMetrics.normalized_discounted_cumulative_gain(gains, perfect)
        return score

    def test_better_ranker_performs_better_than_simple_ranker_on_cranfield_data(self):
        documents = "../data/trec/cranfield/cran.all.1400.xml"
        queries   = "../data/trec/cranfield/cran.qry.xml"
        judgments = "../data/trec/cranfield/cranqrel.trec.txt"
        cranfield = in3120.TREC(documents, queries, judgments)
        analyzer  = in3120.SimpleAnalyzer()
        inverted  = in3120.InMemoryInvertedIndex(cranfield.documents, ["title", "text"], analyzer)
        engine    = in3120.SimpleSearchEngine(cranfield.documents, inverted)
        rankers   = [in3120.BetterRanker(cranfield.documents, inverted), in3120.SimpleRanker()]
        scores    = [self._evaluate_cranfield(cranfield, engine, ranker) for ranker in rankers]
        self.assertGreater(scores[0], scores[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
