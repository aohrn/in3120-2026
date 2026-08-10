# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=protected-access

import unittest
import types
from test_simplesearchengine import TestSimpleSearchEngine
from context import in3120


class TestShingleGenerator(unittest.TestCase):

    def setUp(self):
        self._tokenizer = in3120.ShingleGenerator(3)

    def test_strings(self):
        self.assertListEqual(list(self._tokenizer.strings("")), [])
        self.assertListEqual(list(self._tokenizer.strings("b")), ["b"])
        self.assertListEqual(list(self._tokenizer.strings("ba")), ["ba"])
        self.assertListEqual(list(self._tokenizer.strings("ban")), ["ban"])
        self.assertListEqual(list(self._tokenizer.strings("bana")), ["ban", "ana"])
        self.assertListEqual(list(self._tokenizer.strings("banan")), ["ban", "ana", "nan"])
        self.assertListEqual(list(self._tokenizer.strings("banana")), ["ban", "ana", "nan", "ana"])

    def test_tokens(self):
        self.assertListEqual(list(self._tokenizer.tokens("ba")), [("ba", (0, 2))])
        self.assertListEqual(list(self._tokenizer.tokens("banan")), [("ban", (0, 3)), ("ana", (1, 4)), ("nan", (2, 5))])

    def test_spans(self):
        self.assertListEqual(list(self._tokenizer.spans("ba")), [(0, 2)])
        self.assertListEqual(list(self._tokenizer.spans("banan")), [(0, 3), (1, 4), (2, 5)])

    def test_uses_yield(self):
        for i in range(0, 5):
            text = "x" * i
            self.assertIsInstance(self._tokenizer.spans(text), types.GeneratorType)
            self.assertIsInstance(self._tokenizer.tokens(text), types.GeneratorType)
            self.assertIsInstance(self._tokenizer.strings(text), types.GeneratorType)

    def test_shingled_mesh_corpus(self):
        analyzer = in3120.Analyzer(in3120.SimpleNormalizer(), self._tokenizer)
        corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/mesh.txt"])
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
        engine = in3120.SimpleSearchEngine(corpus, index)
        options = in3120.SimpleSearchEngine.Options(match_threshold=0.1, hit_count=10)
        tester = TestSimpleSearchEngine()
        tester._process_query_verify_matches("orGAnik kEMmistry", engine, options, (10, 8.0, [4408, 4410, 4411, 16980, 16981]))
        tester._process_query_verify_matches("synndrome", engine, options, (10, 7.0, [1275]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
