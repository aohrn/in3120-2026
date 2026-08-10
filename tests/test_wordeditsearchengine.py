# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from context import in3120


class TestWordEditSearchEngine(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        strings = ["foo bar gog", "foo the bar", "alle barna liker boller"]
        self._engine = in3120.WordEditSearchEngine(strings, analyzer)

    def test_simple_lookup_with_known_terms(self):
        options = in3120.EditSearchEngine.Options(upper_bound=1)
        results = list(self._engine.evaluate("bar foo gog", options))
        self.assertEqual(1, len(results))
        self.assertEqual("foo bar gog", results[0].match)
        self.assertEqual(1, results[0].distance)

    def test_simple_lookup_with_an_unknown_term(self):
        options = in3120.EditSearchEngine.Options(upper_bound=1)
        results = list(self._engine.evaluate("alle barna digger boller", options))
        self.assertEqual(1, len(results))
        self.assertEqual("alle barna liker boller", results[0].match)
        self.assertEqual(1, results[0].distance)


if __name__ == '__main__':
    unittest.main(verbosity=2)
