# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestWildcardExpander(unittest.TestCase):

    def setUp(self):
        terms = ["hello", "fishmonger", "filibuster", "fish", "fisher", "sofiafill", "delfi", "banana", "anna"]
        self._expander = in3120.WildcardExpander(terms)

    def test_supported_lookup_keys(self):
        sentinel = self._expander.get_sentinel()
        self.assertListEqual(self._expander.get_keys("X"), [("X" + sentinel, False)])
        self.assertListEqual(self._expander.get_keys("*X"), [("X" + sentinel, True)])
        self.assertListEqual(self._expander.get_keys("X*Y"), [("Y" + sentinel + "X", True)])
        self.assertListEqual(self._expander.get_keys("X*"), [(sentinel + "X", True)])
        self.assertListEqual(self._expander.get_keys("*X*"), [("X", True)])
        self.assertListEqual(self._expander.get_keys("X*Y*Z"), [("Z" + sentinel + "X", True), ("Y", True)])

    def test_unsupported_lookup_keys(self):
        for expression in ["*", "W*X*Y*Z", "X**Y"]:
            with self.assertRaises(KeyError):
                self._expander.get_keys(expression)

    def test_expansion(self):
        self.assertSetEqual(self._expander.expand("dfsd"), set())
        self.assertSetEqual(self._expander.expand("fish"), {"fish"})
        self.assertSetEqual(self._expander.expand("fi*"), {"fish", "fisher", "fishmonger", "filibuster"})
        self.assertSetEqual(self._expander.expand("*fi"), {"delfi"})
        self.assertSetEqual(self._expander.expand("*an*"), {"anna", "banana"})
        self.assertSetEqual(self._expander.expand("fi*er"), {"fisher", "fishmonger", "filibuster"})
        self.assertSetEqual(self._expander.expand("*fi*"), {"fish", "fisher", "fishmonger", "filibuster", "sofiafill", "delfi"})
        self.assertSetEqual(self._expander.expand("fi*mo*er"), {"fishmonger"})
        self.assertSetEqual(self._expander.expand("f*e*"), {"fisher", "fishmonger", "filibuster"})

    def test_unsupported_expansion_patterns(self):
        sentinel = self._expander.get_sentinel()
        for expression in [None, "", sentinel, "AA" + sentinel + "BB"]:
            with self.assertRaises(KeyError):
                self._expander.expand(expression)


if __name__ == '__main__':
    unittest.main(verbosity=2)
