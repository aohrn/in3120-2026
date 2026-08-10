# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from timeit import default_timer as timer
from context import in3120


class TestWindowFinder(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        self._finder = in3120.WindowFinder(analyzer)

    def _scan_and_check(self, buffer: str, query: str, width: int, snippet: str) -> float:
        start = timer()
        result : in3120.WindowFinder.Result = self._finder.scan(buffer, query)
        end = timer()
        self.assertEqual(width, result.width)
        self.assertEqual(snippet, buffer[result.begin:result.end])
        return end - start

    def test_simple_success(self):
        self._scan_and_check("The quality of mercy is not strained", "strained mercy", 4, "mercy is not strained")
        self._scan_and_check("The best pho in the world is this", "world pho", 4, "pho in the world")
        self._scan_and_check("The best pho in the world is this", "best  PHO", 2, "best pho")
        self._scan_and_check("The best  pho in the world is this", "best PHO", 2, "best  pho")
        self._scan_and_check("The best  pho in the world is this", "pho", 1, "pho")
        self._scan_and_check("A D O B E C O D E B A N C", "a b c", 4, "B A N C")
        self._scan_and_check("a", "a", 1, "a")

    def test_simple_failure(self):
        self.assertIsNone(self._finder.scan("The best pho in the world", "banana"))
        self.assertIsNone(self._finder.scan("The best pho in the world", "pho pho"))
        self.assertIsNone(self._finder.scan("pho", "pho pho"))

    def test_scanning_long_buffer_behaves_linearly(self):
        # If processing a buffer of length N takes T seconds, then processing a buffer
        # having length 10N should take approximately 10T seconds.
        factor = 5
        buffer1 = (("a " * 1000) + "X b Y Y ") * 50
        buffer2 = (" " + buffer1) * factor
        buffers = (buffer1, buffer2)
        times = [9999999, 9999999]
        for _ in range(5):
            for i in range(2):
                times[i] = min(times[i], self._scan_and_check(buffers[i], "y x y", 4, "X b Y Y"))
        ratio1 = times[1] / times[0]  # Observed timing ratio.
        ratio2 = ratio1 / factor  # Normalized timing ratio.
        slack = 0.25  # Allow quite a bit of slack, to avoid spurious test failures.
        self.assertLessEqual(ratio2, 1.0 + slack)
        self.assertLessEqual(1.0 - slack, ratio2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
