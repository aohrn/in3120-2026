# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from context import in3120


class TestDummyNormalizer(unittest.TestCase):

    def setUp(self):
        self._normalizer1 = in3120.DummyNormalizer()
        self._normalizer2 = in3120.DummyNormalizer(True)

    def test_without_canonicalization(self):
        self.assertEqual(self._normalizer1.canonicalize("ﾘﾝｸ"), "ﾘﾝｸ")
        self.assertEqual(self._normalizer1.canonicalize("リンク"), "リンク")

    def test_with_canonicalization(self):
        self.assertEqual(self._normalizer2.canonicalize("ﾘﾝｸ"), "リンク")
        self.assertEqual(self._normalizer2.canonicalize("リンク"), "リンク")

    def test_no_normalization(self):
        for normalizer in (self._normalizer1, self._normalizer2):
            self.assertEqual(normalizer.normalize("grÅFustaSJE"), "grÅFustaSJE")


if __name__ == '__main__':
    unittest.main(verbosity=2)
