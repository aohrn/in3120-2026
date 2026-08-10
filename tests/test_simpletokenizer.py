# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from types import GeneratorType
from context import in3120


class TestSimpleTokenizer(unittest.TestCase):

    def setUp(self):
        self._tokenizer = in3120.SimpleTokenizer()

    def test_strings(self):
        result = list(self._tokenizer.strings("Dette  er en\nprøve!"))
        self.assertListEqual(result, ["Dette", "er", "en", "prøve"])

    def test_tokens(self):
        result = list(self._tokenizer.tokens("Dette  er en\nprøve!"))
        self.assertListEqual(result, [("Dette", (0, 5)), ("er", (7, 9)), ("en", (10, 12)), ("prøve", (13, 18))])

    def test_spans(self):
        result = list(self._tokenizer.spans("Dette  er en\nprøve!"))
        self.assertListEqual(result, [(0, 5), (7, 9), (10, 12), (13, 18)])

    def test_join(self):
        tokens = list(self._tokenizer.tokens("Dette  er en\nprøve"))
        tokens.extend([("!", (18, 19)), ("?", (19, 20))])
        result = self._tokenizer.join(iter(tokens))
        self.assertEqual(result, "Dette er en prøve!?")

    def test_empty_input(self):
        self.assertListEqual(list(self._tokenizer.strings("")), [])
        self.assertListEqual(list(self._tokenizer.tokens("")), [])
        self.assertListEqual(list(self._tokenizer.spans("")), [])

    def test_uses_yield(self):
        for i in range(0, 5):
            text = "foo " * i
            self.assertIsInstance(self._tokenizer.spans(text), GeneratorType)
            self.assertIsInstance(self._tokenizer.tokens(text), GeneratorType)
            self.assertIsInstance(self._tokenizer.strings(text), GeneratorType)


if __name__ == '__main__':
    unittest.main(verbosity=2)
