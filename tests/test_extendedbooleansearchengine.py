# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from typing import List, Iterator, Tuple
from context import in3120


class TestExtendedBooleanSearchEngine(unittest.TestCase):

    def setUp(self):
        self._analyzer = in3120.SimpleAnalyzer()
        self._corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/names.txt"])
        self._index = in3120.InMemoryInvertedIndex(self._corpus, ["body"], self._analyzer)
        self._synonyms = in3120.SimpleTrie.from_strings2([("xxxYYYzzz", ["abcdefg", "brock"])], self._analyzer)
        self._engine = in3120.ExtendedBooleanSearchEngine(self._corpus, self._index, self._synonyms)

    def _verify_error(self, expression: str, message: str, options: in3120.BooleanSearchEngine.Options | None):
        results = list(self._engine.evaluate(expression, options))
        self.assertEqual(1, len(results))
        self.assertEqual(results[0].error, message)

    def test_malformed_queries(self):
        for optimize in (True, False):
            options = in3120.BooleanSearchEngine.Options(optimize=optimize)
            self._verify_error("[1, 2, 3]", "Unsupported expression having a body of type List.", options)
            for operator in ("LOOKSLIKE", "SYNONYM", "WILDCARD", "SOUNDSLIKE"):
                self._verify_error(f"{operator}('brocck', 'abba')", f"Operator {operator} expects exactly one argument.", options)
                self._verify_error(f"{operator}(OR('brocck', 'abba'))", f"Operator {operator} has an argument of an unexpected type.", options)
                self._verify_error(f"{operator}('brocck abba')", f"Operator {operator} expects a single-term argument, got ['brocck', 'abba'].", options)

    def _verify_matches(self, expression: str, expected: List[int], options:in3120.BooleanSearchEngine.Options | None):
        results = list(self._engine.evaluate(expression, options))
        self.assertEqual(len(expected), len(results))
        self.assertListEqual(expected, [m.document.document_id for m in results])

    def test_valid_expressions(self):
        for optimize in (True, False):
            options = in3120.BooleanSearchEngine.Options(optimize=optimize)
            self._verify_matches("AND('Mary', LOOKSLIKE('Brocck'))", [20], options)
            self._verify_matches("AND('Mary', WILDCARD('bro*k'))", [20], options)
            self._verify_matches("AND('Mary', WILDCARD('brock'))", [20], options)
            self._verify_matches("AND('Mary', SYNONYM('XXXyyyZZZ'))", [20], options)
            self._verify_matches("AND(SYNONYM('Mary'), SYNONYM('XXXyyyZZZ'))", [20], options)
            self._verify_matches("AND(SOUNDSLIKE('Maery'), OR(SOUNDSLIKE('fdsfsdfsdfsdf'), SOUNDSLIKE('brockh')))", [20], options)
            self._verify_matches("ANDNOT(OR(WILDCARD('mc*dan*el'), SYNONYM('foo')), OR(SOUNDSLIKE('jessikka'), LOOKSLIKE('tracy')))", [2669, 3049], options)

    def test_no_options_specified(self):
        self._verify_matches("AND('Mary', LOOKSLIKE('Brocck'))", [20], None)

    def test_synonym_dictionary_missing_values(self):
        synonyms = in3120.SimpleTrie.from_strings(["a"], self._analyzer)
        with self.assertRaises(AssertionError) as exc:
            in3120.ExtendedBooleanSearchEngine(self._corpus, self._index, synonyms)
        self.assertEqual(str(exc.exception), "Key 'a' has no values.")

    def test_synonym_dictionary_values_not_list(self):
        synonyms = in3120.SimpleTrie.from_strings2([("a", "b")], self._analyzer)
        with self.assertRaises(AssertionError) as exc:
            in3120.ExtendedBooleanSearchEngine(self._corpus, self._index, synonyms)
        self.assertEqual(str(exc.exception), "Key 'a' has meta data that is not a list.")

    def test_synonym_dictionary_key_multiterm(self):
        synonyms = in3120.SimpleTrie.from_strings2([("a b", ["c"])], self._analyzer)
        with self.assertRaises(AssertionError) as exc:
            in3120.ExtendedBooleanSearchEngine(self._corpus, self._index, synonyms)
        self.assertEqual(str(exc.exception), "Key 'a b' maps to a sequence of index terms ['a', 'b'] instead of a single index term.")

    def test_synonym_dictionary_value_multiterm(self):
        synonyms = in3120.SimpleTrie.from_strings2([("a", ["c", "d e"])], self._analyzer)
        with self.assertRaises(AssertionError) as exc:
            in3120.ExtendedBooleanSearchEngine(self._corpus, self._index, synonyms)
        self.assertEqual(str(exc.exception), "At least one of the values for key 'a' maps to a sequence of index terms instead of a single index term.")

    def test_with_unsupported_tokenizer(self):
        class UnsupportedTokenizer(in3120.Tokenizer):
            def spans(self, buffer: str) -> Iterator[Tuple[int, int]]:
                if buffer:
                    if len(buffer) <= 2:
                        yield (0, len(buffer))
                    else:
                        midpoint = len(buffer) // 2
                        yield from ((0, midpoint + 1), (midpoint - 1, len(buffer)))
        for tokenizer in [UnsupportedTokenizer()]:
            analyzer = in3120.Analyzer(in3120.SimpleNormalizer(), tokenizer)
            index = in3120.InMemoryInvertedIndex(self._corpus, ["body"], analyzer)
            synonyms = in3120.SimpleTrie.from_strings2([("xxxYYYzzz", ["abcdefg", "brock"])], self._analyzer)
            with self.assertRaises(AssertionError) as exc:
                in3120.ExtendedBooleanSearchEngine(self._corpus, index, synonyms)
            self.assertEqual(str(exc.exception), "Unsupported tokenization detected.")

    def test_with_unsupported_normalizer(self):
        for normalizer in [in3120.PorterNormalizer(), in3120.SoundexNormalizer()]:
            analyzer = in3120.Analyzer(normalizer, in3120.SimpleTokenizer())
            index = in3120.InMemoryInvertedIndex(self._corpus, ["body"], analyzer)
            synonyms = in3120.SimpleTrie.from_strings2([("xxxYYYzzz", ["abcdefg", "brock"])], self._analyzer)
            with self.assertRaises(AssertionError) as exc:
                in3120.ExtendedBooleanSearchEngine(self._corpus, index, synonyms)
            self.assertEqual(str(exc.exception), "Unsupported normalization detected.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
