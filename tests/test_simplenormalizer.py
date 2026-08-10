# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestSimpleNormalizer(unittest.TestCase):

    def setUp(self):
        self._normalizer = in3120.SimpleNormalizer()

    def test_canonicalize_plain(self):
        self.assertEqual(self._normalizer.canonicalize("Dette ER en\nprØve!"), "Dette ER en\nprØve!")

    def test_canonicalize_japanese_halfwidth_fullwidth(self):
        self.assertEqual(self._normalizer.canonicalize("ﾘﾝｸ"), "リンク")
        self.assertEqual(self._normalizer.canonicalize("リンク"), "リンク")

    def test_canonicalize_latin_capital_letter_c_with_cedilla(self):
        self.assertEqual(self._normalizer.canonicalize("Ç"), "\u00C7")
        self.assertEqual(self._normalizer.canonicalize("\u00C7"), "\u00C7")
        self.assertEqual(self._normalizer.canonicalize("\u0043\u0327"), "\u00C7")

    def test_normalize(self):
        self.assertEqual(self._normalizer.normalize("grÅFustaSJE"), "gråfustasje")


if __name__ == '__main__':
    unittest.main(verbosity=2)
