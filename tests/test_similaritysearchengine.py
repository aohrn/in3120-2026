# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestSimilaritySearchEngine(unittest.TestCase):

    def setUp(self):
        self._analyzer = in3120.SimpleAnalyzer()
        self._corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/docs.json"])
        self._engine = in3120.SimilaritySearchEngine(self._corpus, ["body"], self._analyzer)

    def test_empty_query_yields_no_results(self):
        for empty in ["", " ", None]:
            self.assertEqual(len(list(self._engine.evaluate(empty))), 0)

    def test_client_can_control_number_of_hits(self):
        self.assertEqual(len(list(self._engine.evaluate("search"))), 5)
        options = in3120.SimilaritySearchEngine.Options(hit_count=2)
        self.assertEqual(len(list(self._engine.evaluate("search", options))), 2)
        options = in3120.SimilaritySearchEngine.Options(hit_count=-3)
        self.assertEqual(len(list(self._engine.evaluate("search", options))), 1)

    def test_querying_with_indexed_document_yields_itself_with_perfect_cosine_score(self):
        options = in3120.SimilaritySearchEngine.Options(hit_count=1)
        for document in self._corpus:
            query = document["body"]
            results = list(self._engine.evaluate(query, options))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].document.document_id, document.document_id)
            self.assertEqual(results[0].document["body"], query)
            self.assertAlmostEqual(results[0].score, 1.0, 5)

    def test_empty_corpus_barfs(self):
        for empty in [in3120.InMemoryCorpus(), None]:
            with self.assertRaises(AssertionError):
                in3120.SimilaritySearchEngine(empty, ["body"], self._analyzer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
