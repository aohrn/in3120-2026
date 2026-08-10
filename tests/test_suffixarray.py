# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
import tracemalloc
import inspect
import types
from context import in3120


class TestSuffixArray(unittest.TestCase):

    def setUp(self):
        self._analyzer = in3120.SimpleAnalyzer()

    def _process_query_and_verify_winner(self, engine, query, winners, score):
        options = in3120.SuffixArray.Options(hit_count=5)
        matches = list(engine.evaluate(query, options))
        if winners:
            self.assertGreaterEqual(len(matches), 1)
            self.assertLessEqual(len(matches), 5)
            self.assertIn(matches[0].document.document_id, winners)
            if score:
                self.assertEqual(matches[0].score, score)
        else:
            self.assertEqual(len(matches), 0)

    def test_canonicalized_corpus(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(corpus.size(), {"a": "Japanese リンク"}))
        corpus.add_document(in3120.InMemoryDocument(corpus.size(), {"a": "Cedilla \u0043\u0327 and \u00C7 foo"}))
        engine = in3120.SuffixArray(corpus, ["a"], self._analyzer)
        self._process_query_and_verify_winner(engine, "ﾘﾝｸ", [0], 1)  # Should match "リンク".
        self._process_query_and_verify_winner(engine, "\u00C7", [1], 2)  # Should match "\u0043\u0327", too.

    def test_cran_corpus(self):
        corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/cran.xml"])
        engine = in3120.SuffixArray(corpus, ["body"], self._analyzer)
        self._process_query_and_verify_winner(engine, "visc", [328], 11)
        self._process_query_and_verify_winner(engine, "Of  A", [946], 10)
        self._process_query_and_verify_winner(engine, "", [], None)
        self._process_query_and_verify_winner(engine, "approximate solution", [159, 1374], 3)

    def test_memory_usage(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"a": "o  o\n\n\no\n\no", "b": "o o\no   \no"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"a": "ba", "b": "b bab"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"a": "o  o O o", "b": "o o"}))
        corpus.add_document(in3120.InMemoryDocument(3, {"a": "oO" * 10000, "b": "o"}))
        corpus.add_document(in3120.InMemoryDocument(4, {"a": "cbab o obab O ", "b": "o o " * 10000}))
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        engine = in3120.SuffixArray(corpus, ["a", "b"], self._analyzer)
        self.assertIsNotNone(engine)
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        for statistic in snapshot2.compare_to(snapshot1, "filename"):
            if statistic.traceback[0].filename == inspect.getfile(in3120.SuffixArray):
                message = "Memory usage seems excessive. Are you keeping string copies of all suffixes instead of using offsets, or are you generating too many suffixes by not respecting token boundaries?"
                self.assertLessEqual(statistic.size_diff, 2000000, message)

    def test_multiple_fields(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"field1": "a b c", "field2": "b c d"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"field1": "x", "field2": "y"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"field1": "y", "field2": "z"}))
        engine0 = in3120.SuffixArray(corpus, ["field1", "field2"], self._analyzer)
        engine1 = in3120.SuffixArray(corpus, ["field1"], self._analyzer)
        engine2 = in3120.SuffixArray(corpus, ["field2"], self._analyzer)
        self._process_query_and_verify_winner(engine0, "b c", [0], 2)
        self._process_query_and_verify_winner(engine0, "c b", [], None)  # No cross-field matching.
        self._process_query_and_verify_winner(engine0, "d a", [], None)  # No cross-field matching.
        self._process_query_and_verify_winner(engine0, "y", [1, 2], 1)
        self._process_query_and_verify_winner(engine1, "x", [1], 1)
        self._process_query_and_verify_winner(engine1, "y", [2], 1)
        self._process_query_and_verify_winner(engine1, "z", [], None)
        self._process_query_and_verify_winner(engine2, "z", [2], 1)

    def test_uses_yield(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"a": "the foo bar"}))
        engine = in3120.SuffixArray(corpus, ["a"], self._analyzer)
        matches = engine.evaluate("foo", {})
        self.assertIsInstance(matches, types.GeneratorType, "Are you using yield?")


if __name__ == '__main__':
    unittest.main(verbosity=2)
