# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestChainedNormalizer(unittest.TestCase):

    def setUp(self):
        self._normalizer = in3120.ChainedNormalizer([in3120.PorterNormalizer(), in3120.SoundexNormalizer(), in3120.SimpleNormalizer()])

    def test_normalize(self):
        self.assertEqual(self._normalizer.normalize("OPERations"), "o160")  # OPERations -> oper -> O160 -> o160


if __name__ == '__main__':
    unittest.main(verbosity=2)
