# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestSimpleDAGTrie(unittest.TestCase):

    def setUp(self):
        self._prefix = "thisisshared"
        self._suffix = "alsoshared"
        self._strings = [self._prefix + c + self._suffix for c in "æøå"]
        self._analyzer = in3120.SimpleAnalyzer()
        self._root = in3120.SimpleDAGTrie.from_strings(self._strings, self._analyzer)
        
    def test_containment(self):
        for s in self._strings:
            self.assertTrue(s in self._root)
        for s in (self._prefix + self._suffix, self._prefix, self._suffix, self._prefix + "x" + self._suffix):
            self.assertFalse(s in self._root)

    def test_dump_strings(self):
        self.assertListEqual(list(self._root.strings()), sorted(self._strings))

    def test_counts(self):
        counts = self._root.counts()
        self.assertEqual(counts.node, 1 + len(self._prefix) + len(self._strings) * (1 + len(self._suffix)))
        self.assertEqual(counts.unique, 1 + len(self._prefix) + 1 + len(self._suffix))
        self.assertEqual(counts.final, len(self._strings))
        self.assertEqual(counts.meta, 0)

    def test_to_dot(self):
        dot = self._root.to_dot()
        self.assertIsNotNone(dot)
        self.assertTrue(dot.startswith("digraph {"))
        self.assertEqual(dot.count("shape="), 1 + len(self._prefix) + 1 + len(self._suffix))
        self.assertEqual(dot.count("shape=doublecircle"), 1)
        self.assertEqual(dot.count("shape=circle"), 1 + len(self._prefix) + 1 + len(self._suffix) - 1)
        self.assertEqual(dot.count("->"), len(self._prefix) + len(self._strings) + len(self._suffix))
        self.assertTrue(dot.endswith("}"))

    def test_with_real_corpus(self):
        corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/no.txt"])
        strings = set(t for d in corpus for t, _ in self._analyzer.terms(d["body"]) if len(t) >= 22)
        root = in3120.SimpleDAGTrie.from_strings(strings, self._analyzer)
        counts = root.counts()
        self.assertEqual(counts.final, len(strings))
        self.assertTrue(all(s in root for s in strings))
        dot = root.to_dot()
        self.assertIsNotNone(dot)
        if False:
            with open("../data/no.dot", "w", encoding="utf-8") as f:
                f.write(dot)  # Visualize this!


if __name__ == '__main__':
    unittest.main(verbosity=2)
