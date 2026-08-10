# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestDummyInMemoryInvertedIndex(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"body": "this is a Test"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"body": "test TEST prØve", "foo": "alfa beta gamma"}))
        self._index = in3120.DummyInMemoryInvertedIndex(corpus, ["body", "foo"], analyzer)

    def test_no_posting_lists(self):
        for term in ["hydrogen", "test", "alfa"]:
            self.assertEqual(0, len(list(self._index[term])))

    def test_indexed_terms(self):
        self.assertSetEqual({"alfa", "beta", "gamma", "is", "prøve", "test", "this", "a"}, set(self._index.get_indexed_terms()))

    def test_document_frequencies(self):
        self.assertEqual(2, self._index.get_document_frequency("test"))
        self.assertEqual(1, self._index.get_document_frequency("gamma"))
        self.assertEqual(0, self._index.get_document_frequency("dghgfhsxcxb"))


if __name__ == '__main__':
    unittest.main(verbosity=2)
