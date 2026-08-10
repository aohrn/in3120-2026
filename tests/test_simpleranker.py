# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from context import in3120


class TestSimpleRanker(unittest.TestCase):

    def setUp(self):
        self._ranker = in3120.SimpleRanker()

    def test_term_frequency(self):
        self._ranker.reset(21)
        self._ranker.update("foo", 2, in3120.Posting(21, 4))
        self._ranker.update("bar", 1, in3120.Posting(21, 3))
        self.assertEqual(self._ranker.evaluate(), 11)
        self._ranker.reset(42)
        self._ranker.update("foo", 1, in3120.Posting(42, 1))
        self._ranker.update("baz", 2, in3120.Posting(42, 2))
        self.assertEqual(self._ranker.evaluate(), 5)

    def test_document_id_mismatch(self):
        self._ranker.reset(21)
        with self.assertRaises(AssertionError):
            self._ranker.update("foo", 1, in3120.Posting(42, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
