# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        self._simple = in3120.SimpleAnalyzer()
        self._dummy = in3120.DummyAnalyzer()
        self._custom = in3120.Analyzer(in3120.PorterNormalizer(), in3120.SimpleTokenizer())

    def test_terms_for_dummy_analyzer(self):
        result = [t for t, _ in self._dummy.terms("Dette  er en\nprøve!")]
        self.assertListEqual(result, ["Dette  er en\nprøve!"])

    def test_terms_for_simple_analyzer(self):
        result = [t for t, _ in self._simple.terms("Dette  er en\nprøve!")]
        self.assertListEqual(result, ["dette", "er", "en", "prøve"])

    def test_terms_for_custom_analyzer(self):
        result = [t for t, _ in self._custom.terms("Operations are SIMMERing, baby!")]
        self.assertListEqual(result, ["oper", "are", "simmer", "babi"])

    def test_processed_buffer(self):
        result = self._custom.join("Operations are SIMMERing, baby!")
        self.assertEqual(result, "oper are simmer babi")

    def test_spans_1(self):
        result = list(self._custom.spans("Operations are SIMMERing, baby!"))
        self.assertListEqual(result, [(0, 10), (11, 14), (15, 24), (26, 30)])

    def test_spans_2(self):
        result = [s for _, s in self._custom.terms("Operations are SIMMERing, baby!")]
        self.assertListEqual(result, [(0, 10), (11, 14), (15, 24), (26, 30)])

    def test_can_drop_tokens(self):
        class TokenFilter(in3120.Normalizer):
            def normalize(self, token: str) -> str:
                return "" if token in {"illegal", "pie"} else token
        analyzer = in3120.Analyzer(in3120.ChainedNormalizer([in3120.SimpleNormalizer(), TokenFilter()]), in3120.SimpleTokenizer())
        result = analyzer.join("This banana pie is ILLEGAL!")
        self.assertEqual(result, "this banana is")

    def test_can_access_embedded_normalizer(self):
        for analyzer in (self._simple, self._dummy, self._custom):
            self.assertIsInstance(analyzer.normalizer, in3120.Normalizer)

    def test_can_access_embedded_tokenizer(self):
        for analyzer in (self._simple, self._dummy, self._custom):
            self.assertIsInstance(analyzer.tokenizer, in3120.Tokenizer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
