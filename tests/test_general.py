# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
import sys


class TestGeneral(unittest.TestCase):

    def test_sufficiently_recent_python_version(self):
        # This repository uses language constructs or imports that assume Python >= 3.10.
        self.assertTrue(sys.version_info >= (3, 10), "You need to use at least Python 3.10.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
