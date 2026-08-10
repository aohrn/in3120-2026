# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestSimple9Codec(unittest.TestCase):

    def setUp(self):
        self._pairs = [
            ([], []),
            ([1, 2, 3, 4, 5, 6, 7, 8, 100, 200, 300, 1024], [929383201, 1879150600, 1879355592, 2147484672]),
            ([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1], [265519014]),
            ([3, 2, 3, 3, 2, 3, 1, 2, 2, 2, 3, 2, 3, 1, 0, 2, 3, 3, 3, 3, 1, 1, 1, 2, 2, 1, 3, 3], [398106363, 526999544]),
        ]

    def test_encode(self):
        for decoded, encoded in self._pairs:
            for value in encoded:
                self.assertLessEqual(value, 2**32 - 1)
            self.assertListEqual(list(in3120.Simple9Codec.encode(decoded)), encoded)

    def test_decode(self):
        for decoded, encoded in self._pairs:
            self.assertListEqual(list(in3120.Simple9Codec.decode(iter(encoded))), decoded)

    def test_singleton(self):
        for i in [0, 1, 2, 3, 4, 5, 2**28 - 1]:
            decoded = [i]
            encoded = in3120.Simple9Codec.encode(decoded)
            self.assertListEqual(decoded, list(in3120.Simple9Codec.decode(encoded)))

    def test_negative_number(self):
        decoded = [1, 2, -3, 4]
        with self.assertRaises(ValueError):
            list(in3120.Simple9Codec.encode(decoded))

    def test_too_large_number(self):
        decoded = [1, 2, 2**28, 4]
        with self.assertRaises(ValueError):
            list(in3120.Simple9Codec.encode(decoded))


if __name__ == '__main__':
    unittest.main(verbosity=2)
