# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from context import in3120


class TestBloomFilter(unittest.TestCase):

    def test_add_and_check(self):
        bloom = in3120.BloomFilter(10, 0.05)
        bloom.add(["carrot", "steak", "waffle"])
        self.assertTrue("bazooka" not in bloom)
        self.assertTrue("carrot" in bloom)
        self.assertTrue("steak" in bloom)
        self.assertTrue("waffle" in bloom)
        self.assertTrue("caramel" not in bloom)

    def test_missing_items(self):
        bloom = in3120.BloomFilter()
        for argument in [None, [None], ["foo", None, "bar"]]:
            with self.assertRaises(AssertionError):
                bloom.add(argument)

    def test_valid_parameters(self):
        n, p, m, k = in3120.BloomFilter(1, 0.99).get_parameters()
        self.assertEqual(n, 1)
        self.assertEqual(p, 0.99)
        self.assertEqual(m, 1)
        self.assertEqual(k, 1)
        n, p, m, k = in3120.BloomFilter(10, 0.05).get_parameters()
        self.assertEqual(n, 10)
        self.assertEqual(p, 0.05)
        self.assertEqual(m, 62)
        self.assertEqual(k, 4)
        n, p, m, k = in3120.BloomFilter(4000, 0.0000001).get_parameters()
        self.assertEqual(n, 4000)
        self.assertEqual(p, 0.0000001)
        self.assertEqual(m, 134190)
        self.assertEqual(k, 23)
        n, p, m, k = in3120.BloomFilter(2000, 0.0001).get_parameters()
        self.assertEqual(n, 2000)
        self.assertEqual(p, 0.0001)
        self.assertEqual(m, 38340)
        self.assertEqual(k, 13)

    def test_invalid_parameters(self):
        for n, p in [(0, 0.0001), (1000, 1.0)]:
            with self.assertRaises(AssertionError):
                in3120.BloomFilter(n, p)

    def test_estimated_count(self):
        parameters = [
            (10, 0.05, ("carrot", "steak", "waffle"), 3, 1),
            (100, 0.01, (f"item{i}" for i in range(50)), 50, 5),
            (1000, 0.001, (f"item{i}" for i in range(500)), 500, 10),
        ]
        for n, p, items, expected, delta in parameters:
            bloom = in3120.BloomFilter(n, p)
            bloom.add(items)
            self.assertAlmostEqual(bloom.estimated_count(), expected, delta=delta)


if __name__ == '__main__':
    unittest.main(verbosity=2)
